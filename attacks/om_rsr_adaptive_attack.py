from __future__ import annotations

"""Frozen public-transform-aware attack features for Experiment 16."""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from attacks.mask_aware_spectral_attack import FeatureVector
from communication.orthogonal_spectral_mixing import (
    GLOBAL_MIX_SEED,
    OMPayload,
    OMSchema,
    canonical_group_mask,
    rademacher_signs,
)
from communication.randomized_spectral_release import SPECTRAL_GROUPS

EPSILON = 1e-12
STANDARD_DIMENSION = 24
CROSS_RECON_DIMENSION = 20
MIXED_VALUES_DIMENSION = 84
RELEASE_VALUES_DIMENSION = 28
MASK_DIMENSION = 16
INTERSECTION_DIMENSION = 24
TRANSFORM_METADATA_DIMENSION = 20
PUBLIC_RECON_DIMENSION = 44
FULL_ADAPTIVE_DIMENSION = 168


@dataclass(frozen=True)
class PublicReleaseContext:
    client_id: int
    round_id: int
    release_index: int


def _validate(payloads: Sequence[OMPayload], contexts: Sequence[PublicReleaseContext], schema: OMSchema) -> None:
    if len(payloads) != 2 or len(contexts) != 2:
        raise ValueError("R2 features require exactly two payloads and contexts")
    if float(payloads[0].p) != float(payloads[1].p):
        raise ValueError("release probabilities differ")
    if contexts[0].client_id != contexts[1].client_id:
        raise ValueError("both releases must belong to the same client")
    if contexts[0].round_id >= contexts[1].round_id:
        raise ValueError("release contexts must be chronological")
    if tuple(row.group for row in schema.groups) != tuple(SPECTRAL_GROUPS):
        raise ValueError("public schema order differs")
    for payload in payloads:
        if set(payload.by_group()) != set(SPECTRAL_GROUPS):
            raise ValueError("payload lacks a mixed group")


def _observed(payload: OMPayload, group: str) -> np.ndarray:
    # Undo HT scaling only for values actually carried by the payload.
    return (
        payload.by_group()[group].values.detach().cpu().reshape(-1).numpy()
        .astype(np.float64, copy=True) * float(payload.p)
    )


def _stats(values: np.ndarray, relative_energy: float) -> tuple[float, ...]:
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


def _per_release_blocks(payload: OMPayload) -> dict[str, tuple[float, ...]]:
    observed = {group: _observed(payload, group) for group in SPECTRAL_GROUPS}
    energies = {group: float(np.dot(value, value)) for group, value in observed.items()}
    total = sum(energies.values())
    return {group: _stats(observed[group], energies[group] / (total + EPSILON)) for group in SPECTRAL_GROUPS}


def release_value_features(
    payloads: Sequence[OMPayload], contexts: Sequence[PublicReleaseContext],
    schema: OMSchema, release: int,
) -> FeatureVector:
    _validate(payloads, contexts, schema)
    if release not in (1, 2):
        raise ValueError("release must be 1 or 2")
    block = _per_release_blocks(payloads[release - 1])
    stat_names = ("mean", "population_std", "absolute_mean", "rms", "sign_balance", "second_moment", "relative_group_energy")
    names, values = [], []
    for group in SPECTRAL_GROUPS:
        names.extend(f"release{release}__{group}__{name}" for name in stat_names)
        values.extend(block[group])
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != RELEASE_VALUES_DIMENSION:
        raise RuntimeError("release-only feature dimension changed")
    return result


def mixed_domain_value_features(
    payloads: Sequence[OMPayload], contexts: Sequence[PublicReleaseContext], schema: OMSchema,
) -> FeatureVector:
    _validate(payloads, contexts, schema)
    blocks = [_per_release_blocks(payload) for payload in payloads]
    stat_names = ("mean", "population_std", "absolute_mean", "rms", "sign_balance", "second_moment", "relative_group_energy")
    names, values = [], []
    for release in range(2):
        for group in SPECTRAL_GROUPS:
            names.extend(f"mixed__release{release + 1}__{group}__{name}" for name in stat_names)
            values.extend(blocks[release][group])
    for group in SPECTRAL_GROUPS:
        names.extend(f"mixed__change_r2_minus_r1__{group}__{name}" for name in stat_names)
        values.extend(np.asarray(blocks[1][group]) - np.asarray(blocks[0][group]))
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != MIXED_VALUES_DIMENSION:
        raise RuntimeError("mixed-domain feature dimension changed")
    return result


def _masks(payloads: Sequence[OMPayload], contexts: Sequence[PublicReleaseContext], schema: OMSchema):
    output = []
    for payload, context in zip(payloads, contexts):
        output.append({
            layout.group: canonical_group_mask(
                layout, payload.p, experiment_seed=42, client_id=context.client_id,
                round_id=context.round_id, release_index=context.release_index,
            )[0].detach().cpu().numpy().astype(bool, copy=True)
            for layout in schema.groups
        })
    return output


def public_mask_features(
    payloads: Sequence[OMPayload], contexts: Sequence[PublicReleaseContext], schema: OMSchema,
) -> FeatureVector:
    _validate(payloads, contexts, schema)
    first, second = _masks(payloads, contexts, schema)
    names, values = [], []
    total = sum(layout.D for layout in schema.groups)
    first_count = second_count = intersection_count = union_count = 0
    for layout in schema.groups:
        a, b = first[layout.group], second[layout.group]
        intersection = np.logical_and(a, b)
        union = np.logical_or(a, b)
        names.extend((
            f"mask__{layout.group}__release1_fraction",
            f"mask__{layout.group}__release2_fraction",
            f"mask__{layout.group}__intersection_fraction",
        ))
        values.extend((float(a.mean()), float(b.mean()), float(intersection.mean())))
        first_count += int(a.sum()); second_count += int(b.sum())
        intersection_count += int(intersection.sum()); union_count += int(union.sum())
    names.extend((
        "mask__global__exactly_once_fraction", "mask__global__never_observed_fraction",
        "mask__global__intersection_union_jaccard", "mask__global__signed_visibility_asymmetry",
    ))
    values.extend((
        (union_count - intersection_count) / total,
        (total - union_count) / total,
        intersection_count / union_count if union_count else 0.0,
        (second_count - first_count) / total,
    ))
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != MASK_DIMENSION:
        raise RuntimeError("mask feature dimension changed")
    return result


def _paired(left: np.ndarray, right: np.ndarray) -> tuple[float, ...]:
    if left.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    nl, nr = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    cosine = float(np.dot(left, right) / (nl * nr)) if nl > EPSILON and nr > EPSILON else 0.0
    pearson = 0.0 if left.size < 2 or float(left.std()) <= EPSILON or float(right.std()) <= EPSILON else float(np.corrcoef(left, right)[0, 1])
    delta = right - left
    e1, e2 = float(np.dot(left, left)), float(np.dot(right, right))
    return cosine, pearson, float(np.abs(delta).mean()), float(np.sqrt(np.square(delta).mean())), float(delta.mean()), float((e2 - e1) / (e2 + e1 + EPSILON))


def mixed_intersection_features(
    payloads: Sequence[OMPayload], contexts: Sequence[PublicReleaseContext], schema: OMSchema,
) -> FeatureVector:
    _validate(payloads, contexts, schema)
    first_masks, second_masks = _masks(payloads, contexts, schema)
    names, values = [], []
    stat_names = ("cosine", "pearson", "mean_absolute_change", "rms_change", "signed_mean_change", "symmetric_energy_change")
    for group in SPECTRAL_GROUPS:
        a_mask, b_mask = first_masks[group], second_masks[group]
        a_full = np.full(a_mask.size, np.nan); b_full = np.full(b_mask.size, np.nan)
        a_full[a_mask] = _observed(payloads[0], group)
        b_full[b_mask] = _observed(payloads[1], group)
        both = np.logical_and(a_mask, b_mask)
        names.extend(f"intersection__{group}__{name}" for name in stat_names)
        values.extend(_paired(a_full[both], b_full[both]))
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != INTERSECTION_DIMENSION:
        raise RuntimeError("intersection feature dimension changed")
    return result


def public_transform_metadata_features(
    payloads: Sequence[OMPayload], contexts: Sequence[PublicReleaseContext], schema: OMSchema,
    *, global_mix_seed: int = GLOBAL_MIX_SEED,
) -> FeatureVector:
    _validate(payloads, contexts, schema)
    names, values = [], []
    for layout in schema.groups:
        signs = [rademacher_signs(
            global_mix_seed=global_mix_seed, round_id=context.round_id,
            release_index=context.release_index, group_name=layout.group, D=layout.D,
        )[0].numpy() for context in contexts]
        balance = [float(sign.mean()) for sign in signs]
        names.extend((
            f"transform__{layout.group}__d_over_D",
            f"transform__{layout.group}__padding_fraction",
            f"transform__{layout.group}__release1_sign_balance",
            f"transform__{layout.group}__release2_sign_balance",
            f"transform__{layout.group}__same_sign_fraction",
        ))
        values.extend((layout.d / layout.D, (layout.D - layout.d) / layout.D, balance[0], balance[1], float(np.mean(signs[0] == signs[1]))))
    result = FeatureVector(tuple(names), np.asarray(values, dtype=np.float64))
    if len(result.names) != TRANSFORM_METADATA_DIMENSION:
        raise RuntimeError("transform metadata dimension changed")
    return result


def public_transform_reconstruction_features(standard: np.ndarray, cross_round: np.ndarray) -> FeatureVector:
    standard = np.asarray(standard, dtype=np.float64).reshape(-1)
    cross_round = np.asarray(cross_round, dtype=np.float64).reshape(-1)
    if standard.size != STANDARD_DIMENSION or cross_round.size != CROSS_RECON_DIMENSION:
        raise ValueError("public reconstruction feature input dimension mismatch")
    names = tuple(f"standard__{i:03d}" for i in range(STANDARD_DIMENSION)) + tuple(f"reconstructed_cross_round__{i:03d}" for i in range(CROSS_RECON_DIMENSION))
    result = FeatureVector(names, np.concatenate((standard, cross_round)))
    if len(result.names) != PUBLIC_RECON_DIMENSION:
        raise RuntimeError("public reconstruction dimension changed")
    return result


def full_transform_aware_features(
    standard: np.ndarray, payloads: Sequence[OMPayload],
    contexts: Sequence[PublicReleaseContext], schema: OMSchema,
) -> FeatureVector:
    standard = np.asarray(standard, dtype=np.float64).reshape(-1)
    if standard.size != STANDARD_DIMENSION:
        raise ValueError("standard feature vector must have 24 values")
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
        raise RuntimeError("full adaptive feature schema changed")
    return result


def feature_provenance(names: Sequence[str]) -> list[dict[str, object]]:
    rows = []
    for name in names:
        family = name.split("__", 1)[0]
        rows.append({
            "FeatureName": name,
            "Family": family,
            "Source": "OMR1 retained values and/or deterministic public protocol metadata",
            "UsesRawPrivateUpdate": False,
            "UsesDroppedMixedValue": False,
            "UsesClientIDAsFeature": False,
            "UsesLabelAtInference": False,
            "Allowed": True,
        })
    return rows
