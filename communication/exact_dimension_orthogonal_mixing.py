from __future__ import annotations

"""Exact-dimensional public orthogonal mixing for Experiment 17."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any

import numpy as np
import torch

from communication.implicit_mask_rsr import (
    IMPLICIT_MAGIC,
    PublicMaskContext,
    PublicRSRSchema,
    PublicSchemaEntry,
    decode_implicit_mask_payload,
    encode_implicit_mask_payload,
    schema_from_state_dict,
    transmitted_value_bytes,
)
from communication.randomized_spectral_release import (
    RSRPayload,
    RSREntry,
    SPECTRAL_GROUPS,
    bernoulli_mask,
    tensor_mapping_sha256,
)
from spectral.spectral_groups import GROUPS, inverse_groups, transform_update

GLOBAL_MIX_SEED = 160042
SIGN_NAMESPACE = b"FedSpecShield-ED-OM-Sign-V1\0"
PERMUTATION_NAMESPACE = b"FedSpecShield-ED-OM-Permutation-V1\0"
POLICY_VERSION = "FedSpecShield-ED-OM-RSR-V1"
ED_MAGIC = b"EDM1"
ED_FORMAT = "EDM1"


@dataclass(frozen=True)
class TransformContext:
    global_mix_seed: int
    round_id: int
    release_index: int
    group_name: str


@dataclass(frozen=True)
class StageMetadata:
    stage: int
    seed: int
    leftover_input_slot: int | None
    permutation_sha256: str


@dataclass
class EDRelease:
    payload: RSRPayload
    serialized: bytes
    reconstructed_update: dict[str, torch.Tensor]
    schema: PublicRSRSchema
    mask_rows: list[dict[str, Any]]
    transform_rows: list[dict[str, Any]]
    support: frozenset[str]
    original_coordinates: int
    mixed_coordinates: int
    retained_coordinates: int
    payload_value_bytes: int
    payload_metadata_bytes: int
    forward_mix_ms: float
    inverse_mix_ms: float
    payload_sha256: str
    reconstructed_sha256: str


def stage_count(d: int) -> int:
    if int(d) <= 0:
        raise ValueError("dimension must be positive")
    return int(math.ceil(math.log2(max(int(d), 2))))


def _stable_seed(namespace: bytes, fields: Mapping[str, Any]) -> int:
    encoded = namespace + json.dumps(
        dict(fields), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def derive_sign_seed(context: TransformContext) -> int:
    if context.group_name not in SPECTRAL_GROUPS:
        raise ValueError("unknown spectral group")
    return _stable_seed(SIGN_NAMESPACE, {
        "global_mix_seed": int(context.global_mix_seed),
        "group_name": str(context.group_name),
        "release_index": int(context.release_index),
        "round_id": int(context.round_id),
    })


def derive_stage_seed(context: TransformContext, stage: int) -> int:
    if int(stage) < 0:
        raise ValueError("stage must be nonnegative")
    if context.group_name not in SPECTRAL_GROUPS:
        raise ValueError("unknown spectral group")
    return _stable_seed(PERMUTATION_NAMESPACE, {
        "global_mix_seed": int(context.global_mix_seed),
        "group_name": str(context.group_name),
        "release_index": int(context.release_index),
        "round_id": int(context.round_id),
        "stage": int(stage),
    })


def rademacher_signs(d: int, context: TransformContext, *, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, int]:
    if int(d) <= 0:
        raise ValueError("dimension must be positive")
    seed = derive_sign_seed(context)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (int(d),), generator=generator, dtype=torch.int64)
    return bits.mul(2).sub(1).to(dtype=dtype), seed


def stage_permutations(d: int, context: TransformContext) -> tuple[tuple[torch.Tensor, ...], tuple[StageMetadata, ...]]:
    d = int(d)
    if d <= 0:
        raise ValueError("dimension must be positive")
    permutations: list[torch.Tensor] = []
    metadata: list[StageMetadata] = []
    previous_leftover: int | None = None
    for stage in range(stage_count(d)):
        seed = derive_stage_seed(context, stage)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(d, generator=generator)
        leftover = int(permutation[-1]) if d % 2 else None
        if d > 1 and d % 2 and previous_leftover is not None and leftover == previous_leftover:
            permutation = permutation.clone()
            permutation[0], permutation[-1] = permutation[-1].clone(), permutation[0].clone()
            leftover = int(permutation[-1])
        if d > 1 and d % 2 and previous_leftover is not None and leftover == previous_leftover:
            raise RuntimeError("odd-dimension fairness correction failed")
        digest = hashlib.sha256(permutation.to(torch.int64).numpy().tobytes()).hexdigest()
        permutations.append(permutation)
        metadata.append(StageMetadata(stage, seed, leftover, digest))
        previous_leftover = leftover
    if d > 1 and d % 2 and len({row.leftover_input_slot for row in metadata}) == 1:
        raise RuntimeError("one input slot remained leftover through all stages")
    return tuple(permutations), tuple(metadata)


def _butterfly(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or value.numel() <= 0:
        raise ValueError("butterfly input must be a nonempty vector")
    n = int(value.numel())
    paired = n - (n % 2)
    if paired == 0:
        return value.clone()
    pair = value[:paired].reshape(-1, 2)
    left, right = pair[:, 0], pair[:, 1]
    mixed = torch.stack((left + right, left - right), dim=1) / math.sqrt(2.0)
    return torch.cat((mixed.reshape(-1), value[paired:].clone()))


def exact_orthogonal_mix(value: torch.Tensor, context: TransformContext) -> tuple[torch.Tensor, dict[str, Any]]:
    if value.ndim != 1 or value.numel() <= 0:
        raise ValueError("mix input must be a nonempty vector")
    signs, sign_seed = rademacher_signs(value.numel(), context, dtype=value.dtype)
    permutations, stages = stage_permutations(value.numel(), context)
    out = signs * value
    for permutation in permutations:
        out = _butterfly(out[permutation])
    return out, {
        "Group": context.group_name,
        "Dimension": int(value.numel()),
        "Stages": len(permutations),
        "SignSeed": sign_seed,
        "SignSHA256": hashlib.sha256(signs.numpy().tobytes()).hexdigest(),
        "StageMetadata": [row.__dict__ for row in stages],
        "ClientIDIncluded": False,
    }


def exact_orthogonal_inverse(value: torch.Tensor, context: TransformContext) -> torch.Tensor:
    if value.ndim != 1 or value.numel() <= 0:
        raise ValueError("inverse input must be a nonempty vector")
    permutations, _ = stage_permutations(value.numel(), context)
    out = value.clone()
    for permutation in reversed(permutations):
        out = _butterfly(out)
        inverse = torch.argsort(permutation)
        out = out[inverse]
    signs, _ = rademacher_signs(value.numel(), context, dtype=value.dtype)
    return signs * out


def _spectral_schema_entries(schema: PublicRSRSchema, group: str) -> tuple[PublicSchemaEntry, ...]:
    return tuple(row for row in schema.entries if row.group == group and row.randomized)


def group_dimensions(schema: PublicRSRSchema) -> dict[str, int]:
    result = {group: sum(row.numel for row in _spectral_schema_entries(schema, group)) for group in SPECTRAL_GROUPS}
    if any(value <= 0 for value in result.values()):
        raise ValueError("public schema is missing a spectral group")
    return result


def _flatten_group(groups: Mapping[str, Mapping[str, torch.Tensor]], schema: PublicRSRSchema, group: str) -> torch.Tensor:
    parts = []
    for row in _spectral_schema_entries(schema, group):
        value = groups[group][row.name].detach().cpu().contiguous()
        if tuple(value.shape) != row.shape:
            raise ValueError("group tensor differs from public schema")
        parts.append(value.reshape(-1))
    return torch.cat(parts)


def _restore_group(value: torch.Tensor, schema: PublicRSRSchema, group: str) -> dict[str, torch.Tensor]:
    result = {}; offset = 0
    for row in _spectral_schema_entries(schema, group):
        result[row.name] = value[offset:offset + row.numel].reshape(row.shape).clone()
        offset += row.numel
    if offset != value.numel():
        raise ValueError("mixed vector differs from public schema dimension")
    return result


def serialize_ed_payload(payload: RSRPayload, schema: PublicRSRSchema, context: PublicMaskContext) -> bytes:
    encoded = encode_implicit_mask_payload(payload, schema, context)
    if encoded[:4] != IMPLICIT_MAGIC:
        raise RuntimeError("unexpected implicit serializer magic")
    return ED_MAGIC + encoded[4:]


def deserialize_ed_payload(data: bytes, schema: PublicRSRSchema, context: PublicMaskContext) -> RSRPayload:
    if data[:4] != ED_MAGIC:
        raise ValueError("invalid exact-dimension payload magic")
    return decode_implicit_mask_payload(IMPLICIT_MAGIC + data[4:], schema, context)


def _public_template(groups: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    template = {name: torch.zeros_like(value) for name, value in groups["other"].items()}
    for name in set(groups["graph_low"]) | set(groups["graph_high"]):
        low, high = groups["graph_low"][name], groups["graph_high"][name]
        shape = list(low.shape); shape[0] = low.shape[0] + high.shape[0]
        template[name] = torch.zeros(shape, dtype=low.dtype)
    for name in set(groups["temporal_low"]) | set(groups["temporal_high"]):
        low, high = groups["temporal_low"][name], groups["temporal_high"][name]
        shape = list(low.shape); shape[2] = low.shape[2] + high.shape[2]
        template[name] = torch.zeros(shape, dtype=low.dtype)
    return template


def reconstruct_ed_update(
    payload: RSRPayload, schema: PublicRSRSchema, *,
    experiment_seed: int, client_id: int, round_id: int, release_index: int,
    global_mix_seed: int = GLOBAL_MIX_SEED,
) -> tuple[dict[str, torch.Tensor], float]:
    """Blind reconstruction from payload and deterministic public context only."""
    started = time.perf_counter()
    entries = {(entry.group, entry.name): entry for entry in payload.entries}
    if len(entries) != len(payload.entries):
        raise ValueError("duplicate exact-dimension payload entry")
    expected = {(row.group, row.name) for row in schema.entries}
    if set(entries) != expected:
        raise ValueError("payload entries differ from public schema")
    groups: dict[str, dict[str, torch.Tensor]] = {group: {} for group in GROUPS}
    for row in schema.entries:
        entry = entries[(row.group, row.name)]
        if row.randomized:
            continue
        if entry.indices is not None:
            raise ValueError("dense other tensor unexpectedly sparse")
        groups["other"][row.name] = entry.values.detach().cpu().clone().reshape(row.shape)
    for group in SPECTRAL_GROUPS:
        pieces = []
        for row in _spectral_schema_entries(schema, group):
            entry = entries[(row.group, row.name)]
            if entry.indices is None:
                raise ValueError("spectral entry lacks public canonical mask")
            part = torch.zeros(row.numel, dtype=entry.values.dtype)
            part[entry.indices.detach().cpu().to(torch.int64)] = entry.values.detach().cpu().reshape(-1)
            pieces.append(part)
        y_hat = torch.cat(pieces)
        transform_context = TransformContext(global_mix_seed, round_id, release_index, group)
        z_hat = exact_orthogonal_inverse(y_hat, transform_context)
        groups[group] = _restore_group(z_hat, schema, group)
    update = inverse_groups(groups, _public_template(groups))
    return update, (time.perf_counter() - started) * 1000.0


def exact_dimension_rsr_release(
    update: Mapping[str, torch.Tensor], *, p: float, experiment_seed: int,
    client_id: int, round_id: int, release_index: int,
    global_mix_seed: int = GLOBAL_MIX_SEED,
    schema: PublicRSRSchema | None = None,
) -> EDRelease:
    if not 0.0 < float(p) <= 1.0:
        raise ValueError("p must be in (0,1]")
    public_schema = schema if schema is not None else schema_from_state_dict(update)
    groups = transform_update(update)
    dims = group_dimensions(public_schema)
    entries: list[RSREntry] = []
    mask_rows: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []
    support: set[str] = set()
    total_original = total_mixed = total_retained = 0
    forward_started = time.perf_counter()
    for group in SPECTRAL_GROUPS:
        z = _flatten_group(groups, public_schema, group).to(torch.float32)
        if z.numel() != dims[group]:
            raise RuntimeError("exact-dimension flatten mismatch")
        context = TransformContext(global_mix_seed, round_id, release_index, group)
        y, metadata = exact_orthogonal_mix(z, context)
        if y.numel() != z.numel():
            raise RuntimeError("exact-dimensional transform changed dimension")
        transform_rows.append(metadata)
        offset = 0
        for row in _spectral_schema_entries(public_schema, group):
            segment = y[offset:offset + row.numel]
            template = torch.empty(row.shape, dtype=torch.float32)
            mask, seed = bernoulli_mask(
                template, p, experiment_seed=experiment_seed, client_id=client_id,
                round_id=round_id, release_index=release_index,
                group_name=group, tensor_name=row.name,
            )
            flat_mask = mask.reshape(-1).cpu()
            indices = torch.nonzero(flat_mask, as_tuple=False).flatten().to(torch.int32)
            selected = segment[flat_mask] / float(p)
            entries.append(RSREntry(group, row.name, row.shape, selected, indices))
            mask_rows.append({
                "Group": group, "Tensor": row.name, "MaskSeed": seed,
                "TotalCoordinates": row.numel, "MixedCoordinates": row.numel,
                "RetainedCoordinates": int(indices.numel()),
                "ExtraMaskCoordinates": 0,
            })
            support.update(f"{group}\0{row.name}\0{int(index)}" for index in indices.tolist())
            total_original += row.numel; total_mixed += row.numel; total_retained += int(indices.numel())
            offset += row.numel
        if offset != y.numel():
            raise RuntimeError("group segment accounting mismatch")
    for row in public_schema.entries:
        if row.randomized:
            continue
        value = groups[row.group][row.name].detach().cpu().contiguous()
        if tuple(value.shape) != row.shape:
            raise ValueError("dense other tensor differs from public schema")
        entries.append(RSREntry(row.group, row.name, row.shape, value.clone(), None))
    forward_ms = (time.perf_counter() - forward_started) * 1000.0
    payload = RSRPayload(float(p), tuple(entries))
    public_context = PublicMaskContext(experiment_seed, client_id, round_id, release_index, float(p))
    serialized = serialize_ed_payload(payload, public_schema, public_context)
    decoded = deserialize_ed_payload(serialized, public_schema, public_context)
    reconstructed, inverse_ms = reconstruct_ed_update(
        decoded, public_schema, experiment_seed=experiment_seed, client_id=client_id,
        round_id=round_id, release_index=release_index, global_mix_seed=global_mix_seed,
    )
    value_bytes = transmitted_value_bytes(payload)
    if len(serialized) - value_bytes != 4:
        raise RuntimeError("EDM1 must have exactly four framing bytes")
    return EDRelease(
        payload=payload, serialized=serialized, reconstructed_update=reconstructed,
        schema=public_schema, mask_rows=mask_rows, transform_rows=transform_rows,
        support=frozenset(support), original_coordinates=total_original,
        mixed_coordinates=total_mixed, retained_coordinates=total_retained,
        payload_value_bytes=value_bytes, payload_metadata_bytes=len(serialized)-value_bytes,
        forward_mix_ms=forward_ms, inverse_mix_ms=inverse_ms,
        payload_sha256=hashlib.sha256(serialized).hexdigest(),
        reconstructed_sha256=tensor_mapping_sha256(reconstructed),
    )


def transform_seed_policy() -> dict[str, Any]:
    return {
        "version": POLICY_VERSION,
        "global_mix_seed": GLOBAL_MIX_SEED,
        "sign_namespace_hex": SIGN_NAMESPACE.hex(),
        "permutation_namespace_hex": PERMUTATION_NAMESPACE.hex(),
        "hash": "SHA-256",
        "integer_derivation": "first_8_bytes_big_endian_masked_to_63_bits",
        "sign_fields": ["global_mix_seed", "round_id", "release_index", "group_name"],
        "permutation_fields": ["global_mix_seed", "round_id", "release_index", "group_name", "stage"],
        "client_id_included": False,
        "input_values_included": False,
        "padding": False,
        "stage_rule": "ceil(log2(max(d,2)))",
        "odd_fairness": "consecutive leftover input slots differ for odd d>1",
        "secret": False,
        "signs_uploaded": False,
        "permutations_uploaded": False,
        "dense_matrix_constructed": False,
        "wire_format": ED_FORMAT,
    }
