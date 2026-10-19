from __future__ import annotations

"""Frozen Experiment-15 decomposition features for archived RSR payloads.

Only public masks and actually retained coefficient values are accepted. Dropped
coordinates are represented only by absence and are never assigned a value.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch

from attacks.mask_aware_spectral_attack import EPSILON, FeatureVector
from communication.randomized_spectral_release import RSRPayload, RSREntry, SPECTRAL_GROUPS

GROUP_ORDER = tuple(SPECTRAL_GROUPS)


@dataclass(frozen=True)
class MaskView:
    total: int
    coordinates: frozenset[tuple[str, int]]


def _entries(payload: RSRPayload) -> dict[str, tuple[RSREntry, ...]]:
    grouped: dict[str, list[RSREntry]] = {group: [] for group in GROUP_ORDER}
    seen: set[tuple[str, str]] = set()
    for entry in payload.entries:
        if entry.group not in grouped:
            continue
        key = (entry.group, entry.name)
        if key in seen:
            raise ValueError("duplicate spectral payload entry")
        if entry.indices is None:
            raise ValueError("spectral entry lacks public indices")
        seen.add(key)
        grouped[entry.group].append(entry)
    if any(not grouped[group] for group in GROUP_ORDER):
        raise ValueError("payload is missing a spectral group")
    return {group: tuple(sorted(items, key=lambda x: x.name)) for group, items in grouped.items()}


def mask_views(payload: RSRPayload) -> dict[str, MaskView]:
    """Read shapes/indices only; never access RSREntry.values."""
    result: dict[str, MaskView] = {}
    for group, entries in _entries(payload).items():
        total = 0
        coordinates: set[tuple[str, int]] = set()
        for entry in entries:
            count = int(np.prod(entry.shape, dtype=np.int64))
            total += count
            for index in entry.indices.detach().cpu().to(torch.int64).reshape(-1).tolist():
                if index < 0 or index >= count:
                    raise ValueError("public sparse index is outside tensor shape")
                key = (entry.name, int(index))
                if key in coordinates:
                    raise ValueError("duplicate observed coordinate")
                coordinates.add(key)
        result[group] = MaskView(total, frozenset(coordinates))
    return result


def _selected_observed(payload: RSRPayload, allowed: dict[str, frozenset[tuple[str, int]]]) -> dict[str, dict[tuple[str, int], float]]:
    """Read values only at explicitly allowed, actually transmitted coordinates."""
    if not 0.0 < float(payload.p) <= 1.0:
        raise ValueError("p must be in (0, 1]")
    result: dict[str, dict[tuple[str, int], float]] = {group: {} for group in GROUP_ORDER}
    for group, entries in _entries(payload).items():
        permitted = allowed[group]
        for entry in entries:
            indices = entry.indices.detach().cpu().to(torch.int64).reshape(-1).tolist()
            values = entry.values.detach().cpu().reshape(-1)
            if len(indices) != int(values.numel()):
                raise ValueError("indices and retained values are misaligned")
            for position, index in enumerate(indices):
                key = (entry.name, int(index))
                if key in permitted:
                    result[group][key] = float(values[position].item()) * float(payload.p)
        if set(result[group]) != set(permitted):
            raise ValueError("requested coordinate is not actually transmitted")
    return result


def _two_masks(payloads: Sequence[RSRPayload]) -> tuple[dict[str, MaskView], dict[str, MaskView]]:
    if len(payloads) != 2:
        raise ValueError("R2 decomposition requires exactly two releases")
    first, second = mask_views(payloads[0]), mask_views(payloads[1])
    for group in GROUP_ORDER:
        if first[group].total != second[group].total or first[group].total <= 0:
            raise ValueError("public schema differs across releases")
    return first, second


def release_mask_features(payloads: Sequence[RSRPayload], release: int) -> FeatureVector:
    if len(payloads) != 2 or release not in (1, 2):
        raise ValueError("release must be 1 or 2 for an R2 trajectory")
    view = mask_views(payloads[release - 1])
    names = tuple(f"release{release}_mask__{group}__visible_fraction" for group in GROUP_ORDER)
    values = np.asarray([len(view[group].coordinates) / view[group].total for group in GROUP_ORDER], dtype=np.float64)
    return FeatureVector(names, values)


def _transition_values(a: set | frozenset, b: set | frozenset, total: int) -> tuple[float, ...]:
    intersection, union = a & b, a | b
    return (len(intersection) / total, len(union) / total, len(intersection) / len(union) if union else 0.0, len(a - b) / total, len(b - a) / total, len(a ^ b) / total)


def mask_transition_features(payloads: Sequence[RSRPayload]) -> FeatureVector:
    first, second = _two_masks(payloads)
    stat_names = ("intersection_fraction", "union_fraction", "jaccard_similarity", "release1_only_fraction", "release2_only_fraction", "visibility_flip_fraction")
    names: list[str] = []
    values: list[float] = []
    all_first: set[tuple[str, str, int]] = set()
    all_second: set[tuple[str, str, int]] = set()
    total_all = 0
    for group in GROUP_ORDER:
        a, b, total = first[group].coordinates, second[group].coordinates, first[group].total
        names.extend(f"mask_transition__{group}__{stat}" for stat in stat_names)
        values.extend(_transition_values(a, b, total))
        all_first.update((group, name, index) for name, index in a)
        all_second.update((group, name, index) for name, index in b)
        total_all += total
    names.extend(f"mask_transition__global__{stat}" for stat in stat_names)
    values.extend(_transition_values(all_first, all_second, total_all))
    return FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))


def intersection_structure_features(payloads: Sequence[RSRPayload]) -> FeatureVector:
    first, second = _two_masks(payloads)
    counts = {group: len(first[group].coordinates & second[group].coordinates) for group in GROUP_ORDER}
    total_intersection = sum(counts.values())
    total_coordinates = sum(first[group].total for group in GROUP_ORDER)
    names: list[str] = []
    values: list[float] = []
    for group in GROUP_ORDER:
        names.extend((f"intersection_structure__{group}__intersection_size", f"intersection_structure__{group}__intersection_fraction", f"intersection_structure__{group}__share_of_all_intersections"))
        values.extend((float(counts[group]), counts[group] / first[group].total, counts[group] / total_intersection if total_intersection else 0.0))
    names.extend(("intersection_structure__global__intersection_size", "intersection_structure__global__intersection_fraction"))
    values.extend((float(total_intersection), total_intersection / total_coordinates))
    return FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))


def _distribution(values: list[float] | np.ndarray) -> tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return float(array.mean()), float(array.std(ddof=0)), float(np.abs(array).mean()), float(np.sqrt(np.square(array).mean())), float(((array > 0).sum() - (array < 0).sum()) / array.size)


def release_value_features(payloads: Sequence[RSRPayload], release: int) -> FeatureVector:
    if len(payloads) != 2 or release not in (1, 2):
        raise ValueError("release must be 1 or 2 for an R2 trajectory")
    masks = mask_views(payloads[release - 1])
    observed = _selected_observed(payloads[release - 1], {group: masks[group].coordinates for group in GROUP_ORDER})
    stat_names = ("mean", "population_std", "absolute_mean", "rms", "sign_balance")
    names: list[str] = []
    values: list[float] = []
    for group in GROUP_ORDER:
        names.extend(f"release{release}_values__{group}__{stat}" for stat in stat_names)
        values.extend(_distribution(list(observed[group].values())))
    return FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))


def intersection_observations(payloads: Sequence[RSRPayload]) -> dict[str, tuple[tuple[tuple[str, int], ...], np.ndarray, np.ndarray]]:
    first, second = _two_masks(payloads)
    allowed = {group: frozenset(first[group].coordinates & second[group].coordinates) for group in GROUP_ORDER}
    left, right = _selected_observed(payloads[0], allowed), _selected_observed(payloads[1], allowed)
    result = {}
    for group in GROUP_ORDER:
        keys = tuple(sorted(allowed[group]))
        result[group] = (keys, np.asarray([left[group][k] for k in keys], dtype=np.float64), np.asarray([right[group][k] for k in keys], dtype=np.float64))
    return result


def _paired_stats(left: np.ndarray, right: np.ndarray) -> tuple[float, ...]:
    if left.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    nl, nr = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    cosine = float(np.dot(left, right) / (nl * nr)) if nl > EPSILON and nr > EPSILON else 0.0
    pearson = 0.0 if left.size < 2 or float(left.std(ddof=0)) <= EPSILON or float(right.std(ddof=0)) <= EPSILON else float(np.corrcoef(left, right)[0, 1])
    delta = right - left
    e1, e2 = float(np.dot(left, left)), float(np.dot(right, right))
    return cosine, pearson, float(np.abs(delta).mean()), float(np.sqrt(np.square(delta).mean())), float(delta.mean()), float((e2 - e1) / (e2 + e1 + EPSILON))


def intersection_value_features(payloads: Sequence[RSRPayload]) -> FeatureVector:
    observed = intersection_observations(payloads)
    stats = ("cosine_similarity", "pearson_correlation", "mean_absolute_change", "rms_change", "signed_mean_change", "symmetric_energy_change")
    names: list[str] = []
    values: list[float] = []
    for group in GROUP_ORDER:
        _, left, right = observed[group]
        names.extend(f"intersection_values__{group}__{stat}" for stat in stats)
        values.extend(_paired_stats(left, right))
    return FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))


def xor_observations(payloads: Sequence[RSRPayload]) -> dict[str, dict[str, dict[tuple[str, int], float]]]:
    first, second = _two_masks(payloads)
    a = {group: frozenset(first[group].coordinates - second[group].coordinates) for group in GROUP_ORDER}
    b = {group: frozenset(second[group].coordinates - first[group].coordinates) for group in GROUP_ORDER}
    left, right = _selected_observed(payloads[0], a), _selected_observed(payloads[1], b)
    return {group: {"release1_only": left[group], "release2_only": right[group]} for group in GROUP_ORDER}


def xor_value_features(payloads: Sequence[RSRPayload]) -> FeatureVector:
    observed = xor_observations(payloads)
    stats = ("mean", "absolute_mean", "rms", "sign_balance")
    names: list[str] = []
    values: list[float] = []
    for group in GROUP_ORDER:
        for direction in ("release1_only", "release2_only"):
            dist = _distribution(list(observed[group][direction].values()))
            selected = (dist[0], dist[2], dist[3], dist[4])
            names.extend(f"xor_values__{group}__{direction}__{stat}" for stat in stats)
            values.extend(selected)
    return FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))


def decomposition_feature_provenance(vectors: dict[str, FeatureVector]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, vector in vectors.items():
        uses1 = condition != "Release-2 Mask" and condition != "Release-2 Values"
        uses2 = condition != "Release-1 Mask" and condition != "Release-1 Values"
        uses_mask = "Mask" in condition or condition in {"Intersection Structure", "Intersection Values", "XOR Values"}
        uses_values = condition in {"Release-1 Values", "Release-2 Values", "Intersection Values", "XOR Values"}
        for name in vector.names:
            rows.append({"FeatureName": name, "Condition": condition, "Source": "archived server-visible RSR payload/public schema", "UsesRelease1": uses1, "UsesRelease2": uses2, "UsesMask": uses_mask, "UsesCoefficientValue": uses_values, "UsesIntersection": condition in {"Mask Transition", "Intersection Structure", "Intersection Values"}, "UsesXOR": condition in {"Mask Transition", "XOR Values"}, "UsesRawPrivateUpdate": False, "UsesDroppedCoefficient": False, "UsesPropertyLabelAtInference": False, "Allowed": True})
    return rows
