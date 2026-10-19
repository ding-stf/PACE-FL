from __future__ import annotations

"""Index-free wire representation for the unchanged Randomized Spectral Release.

IMR1 removes only redundant sparse coordinate indices. The mask is public and
is regenerated from the Experiment-11 deterministic seed policy.
"""

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from collections.abc import Mapping

import numpy as np
import torch

from communication.randomized_spectral_release import (
    RSRPayload,
    RSREntry,
    SPECTRAL_GROUPS,
    bernoulli_mask,
)
from spectral.spectral_groups import transform_update

IMPLICIT_MAGIC = b"IMR1"
IMPLICIT_FORMAT = "IMR1"
CANONICAL_ORDER = "entries sorted by (group,name); values flattened row-major/C-contiguous"


@dataclass(frozen=True)
class PublicSchemaEntry:
    group: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    randomized: bool

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


@dataclass(frozen=True)
class PublicRSRSchema:
    entries: tuple[PublicSchemaEntry, ...]

    def to_dict(self) -> dict:
        return {
            "format": IMPLICIT_FORMAT,
            "canonical_order": CANONICAL_ORDER,
            "entries": [asdict(entry) for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "PublicRSRSchema":
        return cls(tuple(
            PublicSchemaEntry(
                group=str(row["group"]),
                name=str(row["name"]),
                shape=tuple(int(x) for x in row["shape"]),
                dtype=str(row["dtype"]),
                randomized=bool(row["randomized"]),
            )
            for row in value["entries"]
        ))


@dataclass(frozen=True)
class PublicMaskContext:
    experiment_seed: int
    client_id: int
    round_id: int
    release_index: int
    p: float


def schema_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> PublicRSRSchema:
    """Derive shape/dtype/order only from the public model state schema."""
    zeros = {name: torch.zeros_like(value, device="cpu") for name, value in state_dict.items()}
    groups = transform_update(zeros)
    rows: list[PublicSchemaEntry] = []
    for group, tensors in groups.items():
        for name, value in tensors.items():
            rows.append(PublicSchemaEntry(
                group=group,
                name=name,
                shape=tuple(int(x) for x in value.shape),
                dtype=str(value.detach().cpu().numpy().dtype),
                randomized=group in SPECTRAL_GROUPS,
            ))
    return PublicRSRSchema(tuple(sorted(rows, key=lambda row: (row.group, row.name))))


def _mask_indices(entry: PublicSchemaEntry, context: PublicMaskContext) -> torch.Tensor:
    template = torch.empty(entry.shape, dtype=torch.from_numpy(
        np.empty((), dtype=np.dtype(entry.dtype))
    ).dtype)
    mask, _ = bernoulli_mask(
        template,
        context.p,
        experiment_seed=context.experiment_seed,
        client_id=context.client_id,
        round_id=context.round_id,
        release_index=context.release_index,
        group_name=entry.group,
        tensor_name=entry.name,
    )
    return torch.nonzero(mask.reshape(-1), as_tuple=False).flatten().to(torch.int32)


def _entry_map(payload: RSRPayload) -> dict[tuple[str, str], RSREntry]:
    result = {(entry.group, entry.name): entry for entry in payload.entries}
    if len(result) != len(payload.entries):
        raise ValueError("duplicate payload entry")
    return result


def encode_implicit_mask_payload(
    payload: RSRPayload,
    schema: PublicRSRSchema,
    context: PublicMaskContext,
) -> bytes:
    """Encode retained values only; validate the archived mask before removal."""
    if float(payload.p) != float(context.p):
        raise ValueError("payload p differs from public context")
    entries = _entry_map(payload)
    if set(entries) != {(row.group, row.name) for row in schema.entries}:
        raise ValueError("payload entries do not match public schema")
    buffer = io.BytesIO()
    buffer.write(IMPLICIT_MAGIC)
    for row in schema.entries:
        entry = entries[(row.group, row.name)]
        if tuple(entry.shape) != row.shape:
            raise ValueError("payload shape differs from public schema")
        values = entry.values.detach().cpu().contiguous()
        if str(values.numpy().dtype) != row.dtype:
            raise ValueError("payload dtype differs from public schema")
        if row.randomized:
            expected_indices = _mask_indices(row, context)
            if entry.indices is None or not torch.equal(
                entry.indices.detach().cpu().to(torch.int32), expected_indices
            ):
                raise ValueError("archived indices differ from regenerated public mask")
            if values.ndim != 1 or values.numel() != expected_indices.numel():
                raise ValueError("retained coefficient count differs from public mask")
        else:
            if entry.indices is not None:
                raise ValueError("dense Other parameter unexpectedly has indices")
            if tuple(values.shape) != row.shape:
                raise ValueError("dense Other parameter shape differs from schema")
        buffer.write(values.numpy().tobytes(order="C"))
    return buffer.getvalue()


def decode_implicit_mask_payload(
    data: bytes,
    schema: PublicRSRSchema,
    context: PublicMaskContext,
) -> RSRPayload:
    """Blind decode using wire bytes and public information only."""
    buffer = io.BytesIO(data)
    if buffer.read(4) != IMPLICIT_MAGIC:
        raise ValueError("invalid implicit-mask RSR payload magic")
    entries: list[RSREntry] = []
    for row in schema.entries:
        dtype = np.dtype(row.dtype)
        if row.randomized:
            indices = _mask_indices(row, context)
            value_count = int(indices.numel())
            shape = (value_count,)
        else:
            indices = None
            value_count = row.numel
            shape = row.shape
        byte_count = value_count * dtype.itemsize
        raw = buffer.read(byte_count)
        if len(raw) != byte_count:
            raise ValueError("truncated implicit-mask RSR payload")
        values = torch.from_numpy(np.frombuffer(raw, dtype=dtype).copy()).reshape(shape)
        entries.append(RSREntry(row.group, row.name, row.shape, values, indices))
    if buffer.read(1):
        raise ValueError("trailing implicit-mask RSR payload bytes")
    return RSRPayload(float(context.p), tuple(entries))


def protocol_minimal_byte_count(
    payload: RSRPayload,
    schema: PublicRSRSchema,
    context: PublicMaskContext,
) -> int:
    """Exact IMR1 byte count: four-byte format id plus transmitted values."""
    encoded = encode_implicit_mask_payload(payload, schema, context)
    return len(encoded)


def logical_payload_sha256(payload: RSRPayload) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(float(payload.p)).encode("ascii") + b"\0")
    for entry in sorted(payload.entries, key=lambda row: (row.group, row.name)):
        digest.update(entry.group.encode("utf-8") + b"\0")
        digest.update(entry.name.encode("utf-8") + b"\0")
        digest.update(json.dumps(list(entry.shape)).encode("ascii") + b"\0")
        values = entry.values.detach().cpu().contiguous()
        digest.update(str(values.numpy().dtype).encode("ascii") + b"\0")
        if entry.indices is None:
            digest.update(b"dense\0")
        else:
            digest.update(b"sparse\0")
            digest.update(entry.indices.detach().cpu().to(torch.int32).contiguous().numpy().tobytes())
        digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def payloads_exact(left: RSRPayload, right: RSRPayload) -> bool:
    if float(left.p) != float(right.p):
        return False
    left_entries = sorted(left.entries, key=lambda row: (row.group, row.name))
    right_entries = sorted(right.entries, key=lambda row: (row.group, row.name))
    if len(left_entries) != len(right_entries):
        return False
    for a, b in zip(left_entries, right_entries):
        if (a.group, a.name, tuple(a.shape)) != (b.group, b.name, tuple(b.shape)):
            return False
        if a.values.dtype != b.values.dtype or not torch.equal(a.values, b.values):
            return False
        if (a.indices is None) != (b.indices is None):
            return False
        if a.indices is not None and not torch.equal(
            a.indices.detach().cpu().to(torch.int32),
            b.indices.detach().cpu().to(torch.int32),
        ):
            return False
    return True


def transmitted_value_bytes(payload: RSRPayload) -> int:
    return sum(
        int(entry.values.numel()) * int(entry.values.element_size())
        for entry in payload.entries
    )
