from __future__ import annotations

"""Frozen Experiment 18 pairing and seed-level aggregation helpers."""

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

from communication.randomized_spectral_release import RSRPayload, SPECTRAL_GROUPS

SEEDS = (42, 123, 2026)
METHODS = ("RSR", "ED-OM-RSR")
SAMPLE_STD_DDOF = 1
LARGE_REVERSE_AUC_GAP = 0.05


def sample_mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("seed-level aggregation requires at least two finite scalar values")
    return float(array.mean()), float(array.std(ddof=SAMPLE_STD_DDOF))


def excess_reduction(rsr_auc_sep: float, ed_auc_sep: float) -> float | None:
    denominator = float(rsr_auc_sep) - 0.5
    if denominator <= 0.0:
        return None
    return 1.0 - (float(ed_auc_sep) - 0.5) / denominator


def _entry_map(payload: RSRPayload):
    result = {(entry.group, entry.name): entry for entry in payload.entries}
    if len(result) != len(payload.entries):
        raise ValueError("duplicate payload entry")
    return result


def mask_signature(payload: RSRPayload) -> str:
    digest = hashlib.sha256()
    for key, entry in sorted(_entry_map(payload).items()):
        if entry.group not in SPECTRAL_GROUPS:
            continue
        if entry.indices is None:
            raise ValueError("spectral entry lacks public mask indices")
        digest.update(entry.group.encode("utf-8") + b"\0")
        digest.update(entry.name.encode("utf-8") + b"\0")
        digest.update(json.dumps(list(entry.shape)).encode("ascii") + b"\0")
        digest.update(entry.indices.detach().cpu().numpy().astype(np.int32, copy=False).tobytes())
    return digest.hexdigest()


def retained_count(payload: RSRPayload) -> int:
    return sum(
        int(entry.indices.numel())
        for entry in payload.entries
        if entry.group in SPECTRAL_GROUPS and entry.indices is not None
    )


def payload_pair_audit(rsr_payload: RSRPayload, ed_payload: RSRPayload) -> dict[str, object]:
    rsr_entries, ed_entries = _entry_map(rsr_payload), _entry_map(ed_payload)
    schema_match = set(rsr_entries) == set(ed_entries) and all(
        tuple(rsr_entries[key].shape) == tuple(ed_entries[key].shape)
        for key in rsr_entries
    )
    mask_match = schema_match
    if schema_match:
        for key, left in rsr_entries.items():
            right = ed_entries[key]
            if left.group in SPECTRAL_GROUPS:
                mask_match = mask_match and left.indices is not None and right.indices is not None
                if mask_match:
                    mask_match = bool(
                        np.array_equal(
                            left.indices.detach().cpu().numpy().astype(np.int32, copy=False),
                            right.indices.detach().cpu().numpy().astype(np.int32, copy=False),
                        )
                    )
    rsr_retained = retained_count(rsr_payload)
    ed_retained = retained_count(ed_payload)
    return {
        "SchemaMatch": bool(schema_match),
        "MaskMatch": bool(mask_match),
        "RSRRetained": rsr_retained,
        "EDRetained": ed_retained,
        "RetainedCountMatch": rsr_retained == ed_retained,
        "RSRMaskHash": mask_signature(rsr_payload),
        "EDMaskHash": mask_signature(ed_payload),
    }


def validate_seed_method_rows(rows: Sequence[Mapping[str, object]]) -> None:
    keys = [(int(row["Seed"]), str(row["Method"])) for row in rows]
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("cross-seed result contamination, duplication, or omission")


def communication_preserved(rsr_bytes: int, ed_bytes: int) -> tuple[bool, float]:
    if int(rsr_bytes) <= 0 or int(ed_bytes) <= 0:
        raise ValueError("serialized byte counts must be positive")
    ratio = int(ed_bytes) / int(rsr_bytes)
    return ratio <= 1.02, ratio


def replication_judgment(
    ed_mean_auc_sep: float,
    paired_reductions: Sequence[float | None],
    improvements: int,
    per_seed_gaps: Sequence[float],
) -> str:
    valid = [float(value) for value in paired_reductions if value is not None]
    if not valid:
        return "NOT REPLICATED"
    mean_reduction = float(np.mean(valid))
    large_reverse = any(float(gap) >= LARGE_REVERSE_AUC_GAP for gap in per_seed_gaps)
    if (
        float(ed_mean_auc_sep) <= 0.58
        and mean_reduction >= 0.50
        and int(improvements) == 3
    ):
        return "STRONGLY REPLICATED"
    if mean_reduction >= 0.50 and int(improvements) >= 2 and not large_reverse:
        return "REPLICATED"
    if (
        0.30 <= mean_reduction < 0.50
        or (int(improvements) >= 2 and mean_reduction > 0.0)
    ):
        return "PARTIALLY REPLICATED"
    return "NOT REPLICATED"

