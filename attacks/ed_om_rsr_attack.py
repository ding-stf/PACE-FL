from __future__ import annotations

"""Frozen transform-aware attack features for exact-dimensional OM-RSR."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from attacks.mask_aware_spectral_attack import FeatureVector
from attacks.om_rsr_adaptive_attack import public_transform_reconstruction_features
from communication.exact_dimension_orthogonal_mixing import (
    GLOBAL_MIX_SEED,
    TransformContext,
    group_dimensions,
    rademacher_signs,
)
from communication.implicit_mask_rsr import PublicRSRSchema
from communication.randomized_spectral_release import RSRPayload, SPECTRAL_GROUPS

EPSILON = 1e-12
STANDARD_DIMENSION = 24
MIXED_VALUES_DIMENSION = 84
RELEASE_VALUES_DIMENSION = 28
MASK_DIMENSION = 16
INTERSECTION_DIMENSION = 24
TRANSFORM_METADATA_DIMENSION = 20
FULL_ADAPTIVE_DIMENSION = 168


@dataclass(frozen=True)
class PublicReleaseContext:
    client_id: int
    round_id: int
    release_index: int


@dataclass(frozen=True)
class GroupView:
    mask: np.ndarray
    observed: np.ndarray
    full_observed: np.ndarray


def _validate(payloads: Sequence[RSRPayload], contexts: Sequence[PublicReleaseContext], schema: PublicRSRSchema) -> None:
    if len(payloads) != 2 or len(contexts) != 2:
        raise ValueError("R2 requires exactly two payloads and contexts")
    if contexts[0].client_id != contexts[1].client_id or contexts[0].round_id >= contexts[1].round_id:
        raise ValueError("contexts must be one client's chronological releases")
    if float(payloads[0].p) != float(payloads[1].p):
        raise ValueError("release probabilities differ")
    group_dimensions(schema)


def _views(payload: RSRPayload, schema: PublicRSRSchema) -> dict[str, GroupView]:
    entries = {(entry.group, entry.name): entry for entry in payload.entries}
    if len(entries) != len(payload.entries):
        raise ValueError("duplicate payload entry")
    result = {}
    for group in SPECTRAL_GROUPS:
        masks, observed, full = [], [], []
        for row in (item for item in schema.entries if item.randomized and item.group == group):
            entry = entries[(row.group, row.name)]
            if entry.indices is None:
                raise ValueError("spectral payload entry lacks public mask")
            mask = np.zeros(row.numel, dtype=bool)
            indices = entry.indices.detach().cpu().numpy().astype(np.int64, copy=False)
            if indices.size and (indices.min() < 0 or indices.max() >= row.numel):
                raise ValueError("public mask coordinate outside schema")
            mask[indices] = True
            values = entry.values.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=True) * float(payload.p)
            if values.size != int(mask.sum()):
                raise ValueError("retained values differ from public mask")
            full_values = np.full(row.numel, np.nan, dtype=np.float64); full_values[mask] = values
            masks.append(mask); observed.append(values); full.append(full_values)
        result[group] = GroupView(np.concatenate(masks), np.concatenate(observed), np.concatenate(full))
    return result


def _statistics(values: np.ndarray, relative_energy: float) -> tuple[float, ...]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(relative_energy)
    second = float(np.square(values).mean())
    return (
        float(values.mean()), float(values.std(ddof=0)), float(np.abs(values).mean()),
        float(np.sqrt(second)),
        float(((values > 0).sum() - (values < 0).sum()) / values.size),
        second, float(relative_energy),
    )


def _release_blocks(payload: RSRPayload, schema: PublicRSRSchema) -> dict[str, tuple[float, ...]]:
    views = _views(payload, schema)
    energies = {group: float(np.dot(view.observed, view.observed)) for group, view in views.items()}
    total = sum(energies.values())
    return {group: _statistics(views[group].observed, energies[group] / (total + EPSILON)) for group in SPECTRAL_GROUPS}


def release_value_features(payloads, contexts, schema, release: int) -> FeatureVector:
    _validate(payloads, contexts, schema)
    if release not in (1, 2):
        raise ValueError("release must be 1 or 2")
    blocks = _release_blocks(payloads[release - 1], schema)
    stats = ("mean", "population_std", "absolute_mean", "rms", "sign_balance", "second_moment", "relative_group_energy")
    names, values = [], []
    for group in SPECTRAL_GROUPS:
        names.extend(f"release{release}__{group}__{name}" for name in stats); values.extend(blocks[group])
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != RELEASE_VALUES_DIMENSION:
        raise RuntimeError("release-only feature dimension changed")
    return result


def mixed_domain_value_features(payloads, contexts, schema) -> FeatureVector:
    _validate(payloads, contexts, schema)
    blocks = [_release_blocks(payload, schema) for payload in payloads]
    stats = ("mean", "population_std", "absolute_mean", "rms", "sign_balance", "second_moment", "relative_group_energy")
    names, values = [], []
    for release in range(2):
        for group in SPECTRAL_GROUPS:
            names.extend(f"mixed__release{release+1}__{group}__{name}" for name in stats); values.extend(blocks[release][group])
    for group in SPECTRAL_GROUPS:
        names.extend(f"mixed__change_r2_minus_r1__{group}__{name}" for name in stats)
        values.extend(np.asarray(blocks[1][group]) - np.asarray(blocks[0][group]))
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != MIXED_VALUES_DIMENSION:
        raise RuntimeError("mixed feature dimension changed")
    return result


def public_mask_features(payloads, contexts, schema) -> FeatureVector:
    _validate(payloads, contexts, schema)
    first, second = (_views(payload, schema) for payload in payloads)
    dims = group_dimensions(schema); total = sum(dims.values())
    names, values = [], []
    first_count = second_count = intersection_count = union_count = 0
    for group in SPECTRAL_GROUPS:
        a, b = first[group].mask, second[group].mask
        intersection, union = np.logical_and(a, b), np.logical_or(a, b)
        names.extend((f"mask__{group}__release1_fraction", f"mask__{group}__release2_fraction", f"mask__{group}__intersection_fraction"))
        values.extend((float(a.mean()), float(b.mean()), float(intersection.mean())))
        first_count += int(a.sum()); second_count += int(b.sum()); intersection_count += int(intersection.sum()); union_count += int(union.sum())
    names.extend(("mask__global__exactly_once_fraction", "mask__global__never_observed_fraction", "mask__global__intersection_union_jaccard", "mask__global__signed_visibility_asymmetry"))
    values.extend(((union_count-intersection_count)/total, (total-union_count)/total, intersection_count/union_count if union_count else 0.0, (second_count-first_count)/total))
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != MASK_DIMENSION:
        raise RuntimeError("mask dimension changed")
    return result


def _paired(left: np.ndarray, right: np.ndarray) -> tuple[float, ...]:
    if left.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    nl, nr = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    cosine = float(np.dot(left, right) / (nl * nr)) if nl > EPSILON and nr > EPSILON else 0.0
    pearson = 0.0 if left.size < 2 or float(left.std()) <= EPSILON or float(right.std()) <= EPSILON else float(np.corrcoef(left, right)[0, 1])
    delta = right - left; e1, e2 = float(np.dot(left, left)), float(np.dot(right, right))
    return cosine, pearson, float(np.abs(delta).mean()), float(np.sqrt(np.square(delta).mean())), float(delta.mean()), float((e2-e1)/(e2+e1+EPSILON))


def mixed_intersection_features(payloads, contexts, schema) -> FeatureVector:
    _validate(payloads, contexts, schema)
    first, second = (_views(payload, schema) for payload in payloads)
    stats = ("cosine", "pearson", "mean_absolute_change", "rms_change", "signed_mean_change", "symmetric_energy_change")
    names, values = [], []
    for group in SPECTRAL_GROUPS:
        both = np.logical_and(first[group].mask, second[group].mask)
        names.extend(f"intersection__{group}__{name}" for name in stats)
        values.extend(_paired(first[group].full_observed[both], second[group].full_observed[both]))
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != INTERSECTION_DIMENSION:
        raise RuntimeError("intersection dimension changed")
    return result


def public_transform_metadata_features(payloads, contexts, schema) -> FeatureVector:
    _validate(payloads, contexts, schema)
    dims = group_dimensions(schema); names, values = [], []
    for group in SPECTRAL_GROUPS:
        signs = []
        for context in contexts:
            transform = TransformContext(GLOBAL_MIX_SEED, context.round_id, context.release_index, group)
            signs.append(rademacher_signs(dims[group], transform)[0].numpy())
        names.extend((
            f"transform__{group}__dimension_retention_ratio",
            f"transform__{group}__padding_fraction",
            f"transform__{group}__release1_sign_balance",
            f"transform__{group}__release2_sign_balance",
            f"transform__{group}__same_sign_fraction",
        ))
        values.extend((1.0, 0.0, float(signs[0].mean()), float(signs[1].mean()), float(np.mean(signs[0] == signs[1]))))
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != TRANSFORM_METADATA_DIMENSION:
        raise RuntimeError("transform metadata dimension changed")
    return result


def full_transform_aware_features(standard, payloads, contexts, schema) -> FeatureVector:
    standard = np.asarray(standard, dtype=np.float64).reshape(-1)
    if standard.size != STANDARD_DIMENSION:
        raise ValueError("standard vector must have 24 values")
    parts = [
        FeatureVector(tuple(f"standard__{i:03d}" for i in range(STANDARD_DIMENSION)), standard),
        mixed_domain_value_features(payloads, contexts, schema),
        public_mask_features(payloads, contexts, schema),
        mixed_intersection_features(payloads, contexts, schema),
        public_transform_metadata_features(payloads, contexts, schema),
    ]
    names = tuple(name for part in parts for name in part.names)
    values = np.concatenate([part.values for part in parts])
    result = FeatureVector(names, values)
    if len(result.names) != FULL_ADAPTIVE_DIMENSION or len(set(result.names)) != FULL_ADAPTIVE_DIMENSION:
        raise RuntimeError("full adaptive schema changed")
    return result


def feature_provenance(names: Sequence[str]) -> list[dict[str, object]]:
    return [{
        "FeatureName": name,
        "Family": name.split("__", 1)[0],
        "Source": "EDM1 retained values and/or deterministic public ED-OM protocol metadata",
        "UsesRawPrivateUpdate": False,
        "UsesDroppedMixedValue": False,
        "UsesClientIDAsFeature": False,
        "UsesLabelAtInference": False,
        "Allowed": True,
    } for name in names]
