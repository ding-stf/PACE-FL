from __future__ import annotations

"""Mechanism-aware features for archived Randomized Spectral Release payloads.

This module is attack-side only. It consumes public masks and retained RSR values;
it has no loader for raw/private updates and never imputes dropped coefficients.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch

from communication.randomized_spectral_release import RSRPayload, RSREntry, SPECTRAL_GROUPS

GROUP_ORDER = tuple(SPECTRAL_GROUPS)
EPSILON = 1e-12
STANDARD_DIMENSION = 24
ADAPTIVE_DIMENSION = 104


@dataclass(frozen=True)
class FeatureVector:
    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.names),):
            raise ValueError("feature names and values are misaligned")
        if not np.isfinite(self.values).all():
            raise ValueError("feature vector contains NaN or infinity")


@dataclass(frozen=True)
class _GroupView:
    total: int
    observed: dict[tuple[str, int], float]
    transmitted: np.ndarray


def recover_observed_original_coefficients(entry: RSREntry, p: float) -> np.ndarray:
    """Recover z=p*tilde_z only at coordinates carried by a sparse RSR entry."""
    if entry.indices is None:
        raise ValueError("observed spectral recovery requires public sparse indices")
    if not 0.0 < float(p) <= 1.0:
        raise ValueError("p must be in (0, 1]")
    values = entry.values.detach().cpu().contiguous().numpy().astype(np.float64, copy=True)
    indices = entry.indices.detach().cpu().reshape(-1).numpy()
    if values.ndim != 1 or values.size != indices.size:
        raise ValueError("indices and retained values must be aligned one-dimensional arrays")
    return values * float(p)


def estimate_group_energy(transmitted_values: np.ndarray, p: float) -> float:
    """Unbiased Horvitz-Thompson estimate p*sum(tilde_z^2)."""
    values = np.asarray(transmitted_values, dtype=np.float64).reshape(-1)
    return float(float(p) * np.dot(values, values))


def estimate_group_mean(transmitted_values: np.ndarray, total_coordinates: int) -> float:
    """Unbiased mean estimate sum(tilde_z)/N."""
    if int(total_coordinates) <= 0:
        return 0.0
    return float(np.asarray(transmitted_values, dtype=np.float64).sum() / int(total_coordinates))


def _spectral_entries(payload: RSRPayload) -> dict[tuple[str, str], RSREntry]:
    rows = {
        (entry.group, entry.name): entry
        for entry in payload.entries
        if entry.group in GROUP_ORDER
    }
    expected = sum(1 for entry in payload.entries if entry.group in GROUP_ORDER)
    if len(rows) != expected:
        raise ValueError("duplicate spectral payload entry")
    if set(group for group, _ in rows) != set(GROUP_ORDER):
        raise ValueError("payload is missing a spectral group")
    return rows


def _group_views(payload: RSRPayload) -> dict[str, _GroupView]:
    rows = _spectral_entries(payload)
    result: dict[str, _GroupView] = {}
    for group in GROUP_ORDER:
        total = 0
        observed: dict[tuple[str, int], float] = {}
        transmitted_parts: list[np.ndarray] = []
        for (_, name), entry in sorted(rows.items()):
            if entry.group != group:
                continue
            if entry.indices is None:
                raise ValueError("spectral entry lacks its public mask coordinates")
            count = int(np.prod(entry.shape, dtype=np.int64))
            total += count
            indices = entry.indices.detach().cpu().to(torch.int64).reshape(-1).numpy()
            if indices.size and (indices.min() < 0 or indices.max() >= count):
                raise ValueError("public sparse index is outside tensor shape")
            original = recover_observed_original_coefficients(entry, payload.p)
            transmitted = entry.values.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=True)
            transmitted_parts.append(transmitted)
            for index, value in zip(indices.tolist(), original.tolist()):
                key = (name, int(index))
                if key in observed:
                    raise ValueError("duplicate observed coordinate")
                observed[key] = float(value)
        result[group] = _GroupView(
            total=total,
            observed=observed,
            transmitted=np.concatenate(transmitted_parts) if transmitted_parts else np.zeros(0, dtype=np.float64),
        )
    return result


def _mask_sets(payload: RSRPayload) -> dict[str, tuple[int, frozenset[tuple[str, int]]]]:
    """Extract only public mask coordinates; coefficient values are never touched."""
    rows = _spectral_entries(payload)
    result: dict[str, tuple[int, frozenset[tuple[str, int]]]] = {}
    for group in GROUP_ORDER:
        total = 0
        coordinates: set[tuple[str, int]] = set()
        for (_, name), entry in sorted(rows.items()):
            if entry.group != group:
                continue
            if entry.indices is None:
                raise ValueError("spectral entry lacks public indices")
            count = int(np.prod(entry.shape, dtype=np.int64))
            total += count
            for index in entry.indices.detach().cpu().to(torch.int64).reshape(-1).tolist():
                if index < 0 or index >= count:
                    raise ValueError("public sparse index is outside tensor shape")
                coordinates.add((name, int(index)))
        result[group] = (total, frozenset(coordinates))
    return result


def compute_mask_only_features(payloads: Sequence[RSRPayload]) -> FeatureVector:
    """Return the frozen 16-dimensional public-mask negative control."""
    if len(payloads) != 2:
        raise ValueError("R2 mask features require exactly two releases")
    first, second = (_mask_sets(payload) for payload in payloads)
    names: list[str] = []
    values: list[float] = []
    all_first: set[tuple[str, str, int]] = set()
    all_second: set[tuple[str, str, int]] = set()
    total_all = 0
    for group in GROUP_ORDER:
        n1, s1 = first[group]
        n2, s2 = second[group]
        if n1 != n2 or n1 <= 0:
            raise ValueError("public schema differs across releases")
        total_all += n1
        all_first.update((group, name, index) for name, index in s1)
        all_second.update((group, name, index) for name, index in s2)
        names.extend([
            f"mask__{group}__release1_visible_fraction",
            f"mask__{group}__release2_visible_fraction",
            f"mask__{group}__intersection_fraction",
        ])
        values.extend([len(s1) / n1, len(s2) / n1, len(s1 & s2) / n1])
    union = all_first | all_second
    intersection = all_first & all_second
    exactly_once = all_first ^ all_second
    names.extend([
        "mask__global__visible_exactly_once_fraction",
        "mask__global__never_visible_fraction",
        "mask__global__intersection_union_jaccard",
        "mask__global__signed_round_visibility_asymmetry",
    ])
    values.extend([
        len(exactly_once) / total_all,
        (total_all - len(union)) / total_all,
        len(intersection) / len(union) if union else 0.0,
        (len(all_second) - len(all_first)) / total_all,
    ])
    return FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))


def _safe_distribution(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(values.mean()),
        float(values.std(ddof=0)),
        float(np.abs(values).mean()),
        float(np.sqrt(np.square(values).mean())),
    )


def _intersection_stats(
    first: dict[tuple[str, int], float],
    second: dict[tuple[str, int], float],
) -> tuple[float, float, float, float, float, float]:
    keys = sorted(set(first) & set(second))
    if not keys:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    left = np.asarray([first[key] for key in keys], dtype=np.float64)
    right = np.asarray([second[key] for key in keys], dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = float(np.dot(left, right) / (left_norm * right_norm)) if left_norm > EPSILON and right_norm > EPSILON else 0.0
    if left.size < 2 or float(left.std(ddof=0)) <= EPSILON or float(right.std(ddof=0)) <= EPSILON:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(left, right)[0, 1])
    delta = right - left
    e1, e2 = float(np.dot(left, left)), float(np.dot(right, right))
    return (
        cosine,
        pearson,
        float(np.abs(delta).mean()),
        float(np.sqrt(np.square(delta).mean())),
        float(delta.mean()),
        float((e2 - e1) / (e2 + e1 + EPSILON)),
    )


def compute_mask_aware_features(payloads: Sequence[RSRPayload]) -> FeatureVector:
    """Return frozen families B-F, excluding the archived standard vector."""
    if len(payloads) != 2:
        raise ValueError("R2 adaptive features require exactly two releases")
    if float(payloads[0].p) != float(payloads[1].p):
        raise ValueError("release probabilities differ")
    p = float(payloads[0].p)
    views = [_group_views(payload) for payload in payloads]
    for group in GROUP_ORDER:
        if views[0][group].total != views[1][group].total:
            raise ValueError("public schema differs across releases")

    names: list[str] = []
    values: list[float] = []

    # B: first-order estimators.
    for group in GROUP_ORDER:
        a, b = views[0][group], views[1][group]
        means = [estimate_group_mean(a.transmitted, a.total), estimate_group_mean(b.transmitted, b.total)]
        pooled = np.asarray(list(a.observed.values()) + list(b.observed.values()), dtype=np.float64)
        if pooled.size:
            sign_balance = float(((pooled > 0).sum() - (pooled < 0).sum()) / pooled.size)
        else:
            sign_balance = 0.0
        names.extend([
            f"first_order__{group}__ht_mean_round_average",
            f"first_order__{group}__ht_mean_signed_change",
            f"first_order__{group}__observed_sign_balance",
        ])
        values.extend([(means[0] + means[1]) / 2.0, means[1] - means[0], sign_balance])

    # C: energy estimators.
    energies = [
        {group: estimate_group_energy(view[group].transmitted, p) for group in GROUP_ORDER}
        for view in views
    ]
    totals = [sum(row.values()) for row in energies]
    for group in GROUP_ORDER:
        e1, e2 = energies[0][group], energies[1][group]
        n = views[0][group].total
        names.extend([
            f"energy__{group}__ht_second_moment_round_average",
            f"energy__{group}__ht_energy_symmetric_change",
            f"energy__{group}__ht_relative_group_energy",
        ])
        values.extend([
            ((e1 / n) + (e2 / n)) / 2.0,
            (e2 - e1) / (abs(e2) + abs(e1) + EPSILON),
            0.5 * (e1 / (totals[0] + EPSILON) + e2 / (totals[1] + EPSILON)),
        ])

    # D: observed-only pooled distributions.
    for group in GROUP_ORDER:
        pooled = np.asarray(
            list(views[0][group].observed.values()) + list(views[1][group].observed.values()),
            dtype=np.float64,
        )
        stats = _safe_distribution(pooled)
        names.extend([
            f"observed__{group}__mean",
            f"observed__{group}__std",
            f"observed__{group}__absolute_mean",
            f"observed__{group}__rms",
        ])
        values.extend(stats)

    # E: cross-round intersection only.
    stat_names = (
        "cosine_similarity", "pearson_correlation", "mean_absolute_change",
        "rms_change", "signed_mean_change", "symmetric_energy_change",
    )
    for group in GROUP_ORDER:
        stats = _intersection_stats(views[0][group].observed, views[1][group].observed)
        names.extend(f"intersection__{group}__{name}" for name in stat_names)
        values.extend(stats)

    mask = compute_mask_only_features(payloads)
    names.extend(mask.names)
    values.extend(mask.values.tolist())
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != ADAPTIVE_DIMENSION - STANDARD_DIMENSION:
        raise RuntimeError(f"unexpected new adaptive feature dimension: {len(result.names)}")
    return result


def build_mask_aware_feature_vector(
    payloads: Sequence[RSRPayload],
    standard_features: np.ndarray,
) -> FeatureVector:
    """Append the frozen B-F features to the unchanged standard RSR vector."""
    standard = np.asarray(standard_features, dtype=np.float64).reshape(-1)
    if standard.shape != (STANDARD_DIMENSION,):
        raise ValueError("standard Experiment-11 spectral vector must have 24 values")
    adaptive = compute_mask_aware_features(payloads)
    names = tuple(f"standard__proposed_spectral_{index:03d}" for index in range(STANDARD_DIMENSION)) + adaptive.names
    values = np.concatenate([standard, adaptive.values])
    result = FeatureVector(names, values)
    if len(result.names) != ADAPTIVE_DIMENSION or len(set(result.names)) != ADAPTIVE_DIMENSION:
        raise RuntimeError("adaptive feature specification dimension/name mismatch")
    return result


def adaptive_feature_provenance(names: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in names:
        family = name.split("__", 1)[0]
        if family == "standard":
            source = "archived standard feature derived from server-visible reconstructed RSR updates"
            mask_required = False
            value_required = True
        elif family == "mask":
            source = "public RSR mask coordinates and public tensor shapes"
            mask_required = True
            value_required = False
        else:
            source = "retained RSR coefficient values, public p, and public mask coordinates"
            mask_required = True
            value_required = True
        rows.append({
            "feature_name": name,
            "source": source,
            "mask_required": mask_required,
            "server_visible_value_required": value_required,
            "raw_private_update_required": False,
            "label_required_at_inference": False,
            "allowed": True,
        })
    return rows
