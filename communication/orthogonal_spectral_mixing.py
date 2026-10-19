from __future__ import annotations

"""Public orthogonal mixing plus canonical RSR sampling for Experiment 16.

The transform and masks are public.  The per-upload OMR1 wire format contains
only dense ``other`` tensors and retained mixed values; masks, signs, and the
Hadamard transform are deterministically reconstructed from public context.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import math
import struct
from typing import Any

import numpy as np
import torch

from communication.randomized_spectral_release import (
    RSREntry,
    SPECTRAL_GROUPS,
    _decode_text,
    _read_exact,
    _serialize_entry,
    bernoulli_mask,
    derive_mask_seed,
    tensor_mapping_sha256,
)
from spectral.spectral_groups import GROUPS, inverse_groups, transform_update

MIX_NAMESPACE = b"FedSpecShield-OM-RSR-Mix-V1\0"
MIX_POLICY_VERSION = "FedSpecShield-OM-RSR-Mix-V1"
WIRE_MAGIC = b"OMR1"
GLOBAL_MIX_SEED = 160042


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


@dataclass(frozen=True)
class GroupLayout:
    group: str
    tensors: tuple[TensorLayout, ...]
    d: int
    D: int
    dtype: str


@dataclass(frozen=True)
class OMSchema:
    groups: tuple[GroupLayout, ...]

    def by_group(self) -> dict[str, GroupLayout]:
        return {row.group: row for row in self.groups}


@dataclass(frozen=True)
class OMGroupObservation:
    group: str
    values: torch.Tensor


@dataclass(frozen=True)
class OMPayload:
    p: float
    dense_entries: tuple[RSREntry, ...]
    groups: tuple[OMGroupObservation, ...]

    def by_group(self) -> dict[str, OMGroupObservation]:
        return {row.group: row for row in self.groups}


@dataclass
class OMRelease:
    payload: OMPayload
    serialized: bytes
    reconstructed_update: dict[str, torch.Tensor]
    schema: OMSchema
    mask_rows: list[dict[str, Any]]
    mix_rows: list[dict[str, Any]]
    support: frozenset[str]
    canonical_spectral_coefficients: int
    padded_mixed_coefficients: int
    released_mixed_coefficients: int
    payload_value_bytes: int
    payload_metadata_bytes: int
    dense_value_bytes: int
    mixed_value_bytes: int
    mix_time_ms: float
    inverse_mix_time_ms: float
    payload_sha256: str
    reconstructed_sha256: str


def next_power_of_two(value: int) -> int:
    if int(value) <= 0:
        raise ValueError("dimension must be positive")
    return 1 << (int(value) - 1).bit_length()


def normalized_fwht(value: torch.Tensor) -> torch.Tensor:
    """Symmetric orthonormal Walsh-Hadamard transform without a dense matrix."""
    if value.ndim != 1:
        raise ValueError("FWHT input must be one-dimensional")
    n = int(value.numel())
    if n <= 0 or n & (n - 1):
        raise ValueError("FWHT length must be a positive power of two")
    out = value.clone()
    width = 1
    while width < n:
        view = out.reshape(-1, 2 * width)
        left = view[:, :width].clone()
        right = view[:, width:].clone()
        out = torch.cat((left + right, left - right), dim=1).reshape(-1)
        width *= 2
    return out / math.sqrt(n)


def derive_mix_seed(
    *, global_mix_seed: int, round_id: int, release_index: int,
    group_name: str, D: int,
) -> int:
    if group_name not in SPECTRAL_GROUPS:
        raise ValueError("unknown spectral group")
    if int(D) <= 0 or int(D) & (int(D) - 1):
        raise ValueError("D must be a positive power of two")
    fields = {
        "D": int(D),
        "global_mix_seed": int(global_mix_seed),
        "group_name": str(group_name),
        "release_index": int(release_index),
        "round_id": int(round_id),
    }
    encoded = MIX_NAMESPACE + json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def rademacher_signs(
    *, global_mix_seed: int, round_id: int, release_index: int,
    group_name: str, D: int, dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    seed = derive_mix_seed(
        global_mix_seed=global_mix_seed, round_id=round_id,
        release_index=release_index, group_name=group_name, D=D,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (int(D),), generator=generator, dtype=torch.int64)
    return (bits.mul(2).sub(1)).to(dtype=dtype), seed


def schema_from_update(update: Mapping[str, torch.Tensor]) -> OMSchema:
    groups = transform_update(update)
    layouts: list[GroupLayout] = []
    for group in SPECTRAL_GROUPS:
        tensors = tuple(
            TensorLayout(name, tuple(int(x) for x in value.shape), str(value.detach().cpu().numpy().dtype))
            for name, value in sorted(groups[group].items())
        )
        if not tensors:
            raise ValueError(f"spectral group {group} is empty")
        dtypes = {row.dtype for row in tensors}
        if len(dtypes) != 1:
            raise ValueError("a mixed group must have one dtype")
        d = sum(row.numel for row in tensors)
        layouts.append(GroupLayout(group, tensors, d, next_power_of_two(d), tensors[0].dtype))
    return OMSchema(tuple(layouts))


def schema_to_dict(schema: OMSchema) -> dict[str, Any]:
    return {
        "group_order": list(SPECTRAL_GROUPS),
        "flatten_order": "tensor_name_UTF8_ascending_then_C_flat_index",
        "groups": [
            {
                "group": row.group, "d": row.d, "D": row.D, "dtype": row.dtype,
                "tensors": [
                    {"name": item.name, "shape": list(item.shape), "dtype": item.dtype,
                     "offset": sum(previous.numel for previous in row.tensors[:index]),
                     "numel": item.numel}
                    for index, item in enumerate(row.tensors)
                ],
            }
            for row in schema.groups
        ],
    }


def schema_from_dict(data: Mapping[str, Any]) -> OMSchema:
    if tuple(data["group_order"]) != tuple(SPECTRAL_GROUPS):
        raise ValueError("public group order mismatch")
    rows = []
    for raw in data["groups"]:
        tensors = tuple(
            TensorLayout(str(item["name"]), tuple(int(x) for x in item["shape"]), str(item["dtype"]))
            for item in raw["tensors"]
        )
        row = GroupLayout(str(raw["group"]), tensors, int(raw["d"]), int(raw["D"]), str(raw["dtype"]))
        if row.d != sum(item.numel for item in tensors) or row.D != next_power_of_two(row.d):
            raise ValueError("invalid public group layout")
        rows.append(row)
    schema = OMSchema(tuple(rows))
    if tuple(row.group for row in schema.groups) != tuple(SPECTRAL_GROUPS):
        raise ValueError("public schema is missing or reorders groups")
    return schema


def _assert_schema(groups: Mapping[str, Mapping[str, torch.Tensor]], schema: OMSchema) -> None:
    for layout in schema.groups:
        actual = tuple(
            TensorLayout(name, tuple(int(x) for x in value.shape), str(value.detach().cpu().numpy().dtype))
            for name, value in sorted(groups[layout.group].items())
        )
        if actual != layout.tensors:
            raise ValueError(f"public architecture schema changed for {layout.group}")


def flatten_group(group: Mapping[str, torch.Tensor], layout: GroupLayout) -> torch.Tensor:
    parts = []
    for item in layout.tensors:
        value = group[item.name].detach().cpu().contiguous()
        if tuple(value.shape) != item.shape:
            raise ValueError("tensor shape differs from public layout")
        parts.append(value.reshape(-1))
    result = torch.cat(parts)
    if result.numel() != layout.d:
        raise RuntimeError("flattened dimension mismatch")
    return result


def pad_vector(value: torch.Tensor, D: int) -> torch.Tensor:
    if value.ndim != 1 or int(value.numel()) > int(D):
        raise ValueError("invalid padding request")
    if int(value.numel()) == int(D):
        return value.clone()
    return torch.cat((value, torch.zeros(int(D) - value.numel(), dtype=value.dtype)))


def restore_group(value: torch.Tensor, layout: GroupLayout) -> dict[str, torch.Tensor]:
    if value.ndim != 1 or value.numel() != layout.d:
        raise ValueError("cropped vector does not match public group layout")
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for item in layout.tensors:
        result[item.name] = value[offset:offset + item.numel].reshape(item.shape).clone()
        offset += item.numel
    return result


def canonical_group_mask(
    layout: GroupLayout, p: float, *, experiment_seed: int, client_id: int,
    round_id: int, release_index: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Concatenate exact legacy masks and extend only numerical padding."""
    pieces: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for item in layout.tensors:
        template = torch.zeros(item.shape, dtype=torch.float32)
        mask, seed = bernoulli_mask(
            template, p, experiment_seed=experiment_seed, client_id=client_id,
            round_id=round_id, release_index=release_index,
            group_name=layout.group, tensor_name=item.name,
        )
        flat = mask.reshape(-1).cpu()
        pieces.append(flat)
        rows.append({
            "Group": layout.group, "Tensor": item.name, "MaskSeed": seed,
            "CanonicalCoordinates": item.numel,
            "CanonicalReleased": int(flat.sum().item()),
        })
    canonical = torch.cat(pieces)
    padding = layout.D - layout.d
    if padding:
        last = layout.tensors[-1]
        seed = derive_mask_seed(
            experiment_seed=experiment_seed, client_id=client_id,
            round_id=round_id, release_index=release_index,
            group_name=layout.group, tensor_name=last.name,
        )
        if float(p) == 1.0:
            extension = torch.ones(padding, dtype=torch.bool)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            draws = torch.rand(last.numel + padding, generator=generator, dtype=torch.float32)
            extension = draws[last.numel:] < float(p)
        rows.append({
            "Group": layout.group, "Tensor": "__padding_continuation__",
            "MaskSeed": seed, "CanonicalCoordinates": 0,
            "CanonicalReleased": 0, "PaddingCoordinates": padding,
            "PaddingReleased": int(extension.sum().item()),
        })
        mask = torch.cat((canonical, extension))
    else:
        mask = canonical
    if mask.numel() != layout.D:
        raise RuntimeError("mixed mask dimension mismatch")
    return mask, rows


def _decode_entry(buffer: io.BytesIO) -> RSREntry:
    group, name = _decode_text(buffer), _decode_text(buffer)
    sparse = bool(struct.unpack("<B", _read_exact(buffer, 1))[0])
    if sparse:
        raise ValueError("OM dense entry unexpectedly marked sparse")
    ndim = struct.unpack("<B", _read_exact(buffer, 1))[0]
    shape = tuple(struct.unpack(f"<{ndim}I", _read_exact(buffer, 4 * ndim))) if ndim else ()
    dtype = np.dtype(_decode_text(buffer))
    byte_count = struct.unpack("<Q", _read_exact(buffer, 8))[0]
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if byte_count != expected:
        raise ValueError("dense tensor byte count does not match shape")
    values = torch.from_numpy(np.frombuffer(_read_exact(buffer, byte_count), dtype=dtype).copy()).reshape(shape)
    return RSREntry(group, name, shape, values, None)


def serialize_om_payload(payload: OMPayload) -> bytes:
    if tuple(row.group for row in payload.groups) != tuple(SPECTRAL_GROUPS):
        raise ValueError("OM payload groups must use fixed public order")
    buffer = io.BytesIO()
    buffer.write(WIRE_MAGIC)
    buffer.write(struct.pack("<dH", float(payload.p), len(payload.dense_entries)))
    for entry in sorted(payload.dense_entries, key=lambda row: (row.group, row.name)):
        if entry.group != "other" or entry.indices is not None:
            raise ValueError("only dense other entries are allowed")
        buffer.write(_serialize_entry(entry))
    buffer.write(struct.pack("<B", len(payload.groups)))
    for row in payload.groups:
        values = row.values.detach().cpu().to(torch.float32).contiguous().numpy()
        if values.ndim != 1:
            raise ValueError("mixed observed values must be flat")
        buffer.write(struct.pack("<Q", values.size))
        buffer.write(values.tobytes(order="C"))
    return buffer.getvalue()


def deserialize_om_payload(
    data: bytes, schema: OMSchema, *, experiment_seed: int, client_id: int,
    round_id: int, release_index: int,
) -> OMPayload:
    buffer = io.BytesIO(data)
    if _read_exact(buffer, 4) != WIRE_MAGIC:
        raise ValueError("invalid OM payload magic")
    p, dense_count = struct.unpack("<dH", _read_exact(buffer, 10))
    dense = tuple(_decode_entry(buffer) for _ in range(dense_count))
    group_count = struct.unpack("<B", _read_exact(buffer, 1))[0]
    if group_count != len(SPECTRAL_GROUPS):
        raise ValueError("OM payload group count mismatch")
    observations = []
    for layout in schema.groups:
        count = struct.unpack("<Q", _read_exact(buffer, 8))[0]
        mask, _ = canonical_group_mask(
            layout, p, experiment_seed=experiment_seed, client_id=client_id,
            round_id=round_id, release_index=release_index,
        )
        expected = int(mask.sum().item())
        if count != expected:
            raise ValueError("retained mixed-value count disagrees with public mask")
        raw = np.frombuffer(_read_exact(buffer, count * 4), dtype=np.float32).copy()
        observations.append(OMGroupObservation(layout.group, torch.from_numpy(raw)))
    if buffer.read(1):
        raise ValueError("trailing OM payload bytes")
    return OMPayload(float(p), dense, tuple(observations))


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


def reconstruct_om_update(
    payload: OMPayload, schema: OMSchema, *, experiment_seed: int,
    client_id: int, round_id: int, release_index: int,
    global_mix_seed: int = GLOBAL_MIX_SEED,
) -> tuple[dict[str, torch.Tensor], float]:
    """Reconstruct using only payload plus deterministic public metadata."""
    import time
    started = time.perf_counter()
    groups: dict[str, dict[str, torch.Tensor]] = {group: {} for group in GROUPS}
    for entry in payload.dense_entries:
        if entry.group != "other" or entry.indices is not None:
            raise ValueError("invalid dense OM entry")
        groups["other"][entry.name] = entry.values.detach().cpu().clone().reshape(entry.shape)
    observations = payload.by_group()
    if set(observations) != set(SPECTRAL_GROUPS):
        raise ValueError("payload does not contain the fixed spectral groups")
    for layout in schema.groups:
        mask, _ = canonical_group_mask(
            layout, payload.p, experiment_seed=experiment_seed, client_id=client_id,
            round_id=round_id, release_index=release_index,
        )
        selected = observations[layout.group].values.detach().cpu().to(torch.float32).reshape(-1)
        if selected.numel() != int(mask.sum().item()):
            raise ValueError("payload values and public mask disagree")
        y_hat = torch.zeros(layout.D, dtype=selected.dtype)
        y_hat[mask] = selected
        signs, _ = rademacher_signs(
            global_mix_seed=global_mix_seed, round_id=round_id,
            release_index=release_index, group_name=layout.group,
            D=layout.D, dtype=y_hat.dtype,
        )
        z_hat = signs * normalized_fwht(y_hat)
        groups[layout.group] = restore_group(z_hat[:layout.d], layout)
    update = inverse_groups(groups, _public_template(groups))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return update, elapsed_ms


def orthogonally_mixed_rsr(
    update: Mapping[str, torch.Tensor], *, p: float, experiment_seed: int,
    client_id: int, round_id: int, release_index: int,
    global_mix_seed: int = GLOBAL_MIX_SEED, schema: OMSchema | None = None,
) -> OMRelease:
    import time
    if not 0.0 < float(p) <= 1.0:
        raise ValueError("p must be in (0, 1]")
    groups = transform_update(update)
    public_schema = schema if schema is not None else schema_from_update(update)
    _assert_schema(groups, public_schema)
    dense_entries = tuple(
        RSREntry("other", name, tuple(int(x) for x in value.shape), value.detach().cpu().clone(), None)
        for name, value in sorted(groups["other"].items())
    )
    observations: list[OMGroupObservation] = []
    mask_rows: list[dict[str, Any]] = []
    mix_rows: list[dict[str, Any]] = []
    support: set[str] = set()
    canonical_total = padded_total = released_total = 0
    mixed_value_bytes = 0
    mix_started = time.perf_counter()
    for layout in public_schema.groups:
        z = flatten_group(groups[layout.group], layout).to(torch.float32)
        z_pad = pad_vector(z, layout.D)
        signs, mix_seed = rademacher_signs(
            global_mix_seed=global_mix_seed, round_id=round_id,
            release_index=release_index, group_name=layout.group,
            D=layout.D, dtype=z_pad.dtype,
        )
        y = normalized_fwht(signs * z_pad)
        mask, rows = canonical_group_mask(
            layout, p, experiment_seed=experiment_seed, client_id=client_id,
            round_id=round_id, release_index=release_index,
        )
        selected = y[mask] / float(p)
        observations.append(OMGroupObservation(layout.group, selected))
        mask_rows.extend(rows)
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        support.update(f"{layout.group}\0{int(index)}" for index in indices)
        canonical_total += layout.d
        padded_total += layout.D
        released_total += selected.numel()
        mixed_value_bytes += selected.numel() * selected.element_size()
        mix_rows.append({
            "Group": layout.group, "d": layout.d, "D": layout.D,
            "PaddingCoordinates": layout.D - layout.d,
            "MixSeed": mix_seed, "ReleasedMixedCoordinates": int(selected.numel()),
        })
    mix_time_ms = (time.perf_counter() - mix_started) * 1000.0
    payload = OMPayload(float(p), dense_entries, tuple(observations))
    serialized = serialize_om_payload(payload)
    decoded = deserialize_om_payload(
        serialized, public_schema, experiment_seed=experiment_seed,
        client_id=client_id, round_id=round_id, release_index=release_index,
    )
    reconstructed, inverse_ms = reconstruct_om_update(
        decoded, public_schema, experiment_seed=experiment_seed,
        client_id=client_id, round_id=round_id, release_index=release_index,
        global_mix_seed=global_mix_seed,
    )
    dense_value_bytes = sum(entry.values.numel() * entry.values.element_size() for entry in dense_entries)
    value_bytes = dense_value_bytes + mixed_value_bytes
    return OMRelease(
        payload=payload, serialized=serialized, reconstructed_update=reconstructed,
        schema=public_schema, mask_rows=mask_rows, mix_rows=mix_rows,
        support=frozenset(support), canonical_spectral_coefficients=canonical_total,
        padded_mixed_coefficients=padded_total, released_mixed_coefficients=released_total,
        payload_value_bytes=value_bytes, payload_metadata_bytes=len(serialized) - value_bytes,
        dense_value_bytes=dense_value_bytes, mixed_value_bytes=mixed_value_bytes,
        mix_time_ms=mix_time_ms, inverse_mix_time_ms=inverse_ms,
        payload_sha256=hashlib.sha256(serialized).hexdigest(),
        reconstructed_sha256=tensor_mapping_sha256(reconstructed),
    )


def mixing_seed_policy() -> dict[str, Any]:
    return {
        "version": MIX_POLICY_VERSION,
        "namespace_hex": MIX_NAMESPACE.hex(),
        "global_mix_seed": GLOBAL_MIX_SEED,
        "hash": "SHA-256",
        "integer_derivation": "first_8_bytes_big_endian_masked_to_63_bits",
        "fields": ["global_mix_seed", "round_id", "release_index", "group_name", "D"],
        "client_id_included": False,
        "data_dependent": False,
        "label_dependent": False,
        "secret": False,
        "matrix_uploaded": False,
        "sign_vector_uploaded": False,
    }
