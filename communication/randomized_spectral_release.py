from __future__ import annotations

"""Deterministic non-private randomized spectral release for Experiment 11."""

from dataclasses import dataclass
import hashlib
import io
import json
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from spectral.spectral_groups import GROUPS, inverse_groups, transform_update

SPECTRAL_GROUPS = ("graph_low", "graph_high", "temporal_low", "temporal_high")
MASK_NAMESPACE = b"FedSpecShield-RSR-Mask-V1\0"
MASK_POLICY_VERSION = "FedSpecShield-RSR-Mask-V1"


@dataclass(frozen=True)
class RSREntry:
    group: str
    name: str
    shape: tuple[int, ...]
    values: torch.Tensor
    indices: torch.Tensor | None


@dataclass(frozen=True)
class RSRPayload:
    p: float
    entries: tuple[RSREntry, ...]


@dataclass
class RSRRelease:
    payload: RSRPayload
    serialized: bytes
    reconstructed_update: dict[str, torch.Tensor]
    mask_rows: list[dict[str, Any]]
    support: frozenset[str]
    total_spectral_coefficients: int
    released_spectral_coefficients: int
    spectral_entry_bytes: int
    payload_sha256: str
    reconstructed_sha256: str


def derive_mask_seed(
    *, experiment_seed: int, client_id: int, round_id: int, release_index: int,
    group_name: str, tensor_name: str,
) -> int:
    """Derive a stable mask seed without inspecting values or private labels."""
    payload = {
        "experiment_seed": int(experiment_seed), "client_id": int(client_id),
        "round_id": int(round_id), "release_index": int(release_index),
        "group_name": str(group_name), "tensor_name": str(tensor_name),
    }
    encoded = MASK_NAMESPACE + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def bernoulli_mask(
    value: torch.Tensor, p: float, *, experiment_seed: int, client_id: int,
    round_id: int, release_index: int, group_name: str, tensor_name: str,
) -> tuple[torch.Tensor, int]:
    if not 0.0 < float(p) <= 1.0:
        raise ValueError("p must be in (0, 1]")
    seed = derive_mask_seed(
        experiment_seed=experiment_seed, client_id=client_id, round_id=round_id,
        release_index=release_index, group_name=group_name, tensor_name=tensor_name,
    )
    if float(p) == 1.0:
        return torch.ones_like(value, dtype=torch.bool), seed
    generator = torch.Generator(device=value.device).manual_seed(seed)
    mask = torch.rand(
        value.shape, generator=generator, device=value.device, dtype=torch.float32
    ) < float(p)
    return mask, seed


def _encode_text(buffer: io.BytesIO, value: str) -> None:
    encoded = value.encode("utf-8")
    if len(encoded) > 65535:
        raise ValueError("payload identifier is too long")
    buffer.write(struct.pack("<H", len(encoded)))
    buffer.write(encoded)


def _read_exact(buffer: io.BytesIO, count: int) -> bytes:
    value = buffer.read(count)
    if len(value) != count:
        raise ValueError("truncated RSR payload")
    return value


def _decode_text(buffer: io.BytesIO) -> str:
    count = struct.unpack("<H", _read_exact(buffer, 2))[0]
    return _read_exact(buffer, count).decode("utf-8")


def _serialize_entry(entry: RSREntry) -> bytes:
    buffer = io.BytesIO()
    _encode_text(buffer, entry.group)
    _encode_text(buffer, entry.name)
    sparse = entry.indices is not None
    buffer.write(struct.pack("<B", int(sparse)))
    shape = tuple(int(x) for x in entry.shape)
    buffer.write(struct.pack("<B", len(shape)))
    if shape:
        buffer.write(struct.pack(f"<{len(shape)}I", *shape))
    values = entry.values.detach().cpu().contiguous().numpy()
    _encode_text(buffer, str(values.dtype))
    if sparse:
        indices = entry.indices.detach().cpu().to(torch.int32).contiguous().numpy()
        if indices.ndim != 1 or values.ndim != 1 or len(indices) != len(values):
            raise ValueError("sparse RSR entries require aligned flat indices and values")
        buffer.write(struct.pack("<Q", len(indices)))
        buffer.write(indices.tobytes(order="C"))
        buffer.write(values.tobytes(order="C"))
    else:
        raw = values.tobytes(order="C")
        buffer.write(struct.pack("<Q", len(raw)))
        buffer.write(raw)
    return buffer.getvalue()


def serialize_rsr_payload(payload: RSRPayload) -> bytes:
    buffer = io.BytesIO()
    buffer.write(b"RSR1")
    buffer.write(struct.pack("<dH", float(payload.p), len(payload.entries)))
    for entry in sorted(payload.entries, key=lambda row: (row.group, row.name)):
        buffer.write(_serialize_entry(entry))
    return buffer.getvalue()


def deserialize_rsr_payload(data: bytes) -> RSRPayload:
    buffer = io.BytesIO(data)
    if _read_exact(buffer, 4) != b"RSR1":
        raise ValueError("invalid RSR payload magic")
    p, count = struct.unpack("<dH", _read_exact(buffer, 10))
    entries: list[RSREntry] = []
    for _ in range(count):
        group, name = _decode_text(buffer), _decode_text(buffer)
        sparse = bool(struct.unpack("<B", _read_exact(buffer, 1))[0])
        ndim = struct.unpack("<B", _read_exact(buffer, 1))[0]
        shape = tuple(struct.unpack(f"<{ndim}I", _read_exact(buffer, 4 * ndim))) if ndim else ()
        dtype = np.dtype(_decode_text(buffer))
        if sparse:
            item_count = struct.unpack("<Q", _read_exact(buffer, 8))[0]
            indices = torch.from_numpy(
                np.frombuffer(_read_exact(buffer, item_count * 4), dtype=np.int32).copy()
            )
            values = torch.from_numpy(
                np.frombuffer(_read_exact(buffer, item_count * dtype.itemsize), dtype=dtype).copy()
            )
        else:
            byte_count = struct.unpack("<Q", _read_exact(buffer, 8))[0]
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if byte_count != expected:
                raise ValueError("dense RSR tensor byte count does not match shape")
            indices = None
            values = torch.from_numpy(
                np.frombuffer(_read_exact(buffer, byte_count), dtype=dtype).copy()
            ).reshape(shape)
        entries.append(RSREntry(group, name, shape, values, indices))
    if buffer.read(1):
        raise ValueError("trailing RSR payload bytes")
    return RSRPayload(float(p), tuple(entries))


def payload_to_groups(payload: RSRPayload) -> dict[str, dict[str, torch.Tensor]]:
    groups: dict[str, dict[str, torch.Tensor]] = {group: {} for group in GROUPS}
    for entry in payload.entries:
        if entry.group not in groups or entry.name in groups[entry.group]:
            raise ValueError("unknown group or duplicate RSR tensor entry")
        if entry.indices is None:
            value = entry.values.detach().cpu().clone().reshape(entry.shape)
        else:
            value = torch.zeros(int(np.prod(entry.shape, dtype=np.int64)), dtype=entry.values.dtype)
            value[entry.indices.to(torch.int64)] = entry.values.detach().cpu()
            value = value.reshape(entry.shape)
        groups[entry.group][entry.name] = value
    return groups


def _public_template(groups: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    template = {name: torch.zeros_like(value) for name, value in groups["other"].items()}
    for name in set(groups["graph_low"]) | set(groups["graph_high"]):
        low, high = groups["graph_low"][name], groups["graph_high"][name]
        shape = list(low.shape)
        shape[0] = low.shape[0] + high.shape[0]
        template[name] = torch.zeros(shape, dtype=low.dtype)
    for name in set(groups["temporal_low"]) | set(groups["temporal_high"]):
        low, high = groups["temporal_low"][name], groups["temporal_high"][name]
        shape = list(low.shape)
        shape[2] = low.shape[2] + high.shape[2]
        template[name] = torch.zeros(shape, dtype=low.dtype)
    return template


def reconstruct_rsr_update(payload: RSRPayload) -> dict[str, torch.Tensor]:
    groups = payload_to_groups(payload)
    return inverse_groups(groups, _public_template(groups))


def tensor_mapping_sha256(update: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(update):
        value = update[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def randomized_spectral_release(
    update: Mapping[str, torch.Tensor], *, p: float, experiment_seed: int,
    client_id: int, round_id: int, release_index: int,
) -> RSRRelease:
    """Create a public sparse payload and reconstruct exactly what the server uses."""
    groups = transform_update(update)
    entries: list[RSREntry] = []
    mask_rows: list[dict[str, Any]] = []
    support: set[str] = set()
    total_spectral = released_spectral = 0
    for group in GROUPS:
        for name, value in sorted(groups[group].items()):
            shape = tuple(int(x) for x in value.shape)
            if group in SPECTRAL_GROUPS:
                mask, seed = bernoulli_mask(
                    value, p, experiment_seed=experiment_seed, client_id=client_id,
                    round_id=round_id, release_index=release_index,
                    group_name=group, tensor_name=name,
                )
                flat_mask = mask.reshape(-1)
                indices = torch.nonzero(flat_mask, as_tuple=False).flatten().to(torch.int32)
                selected = value.reshape(-1)[flat_mask] / float(p)
                total, released = int(value.numel()), int(indices.numel())
                total_spectral += total
                released_spectral += released
                support.update(
                    f"{group}\0{name}\0{int(index)}"
                    for index in indices.detach().cpu().tolist()
                )
                entries.append(RSREntry(group, name, shape, selected, indices))
                mask_rows.append({
                    "Group": group, "Tensor": name, "MaskSeed": seed,
                    "TotalCoefficients": total, "ReleasedCoefficients": released,
                    "RealizedReleaseRatio": released / total if total else 0.0,
                })
            else:
                entries.append(RSREntry(group, name, shape, value.detach().cpu().clone(), None))
    payload = RSRPayload(float(p), tuple(entries))
    serialized = serialize_rsr_payload(payload)
    reconstructed = reconstruct_rsr_update(deserialize_rsr_payload(serialized))
    spectral_entry_bytes = sum(
        len(_serialize_entry(entry)) for entry in payload.entries
        if entry.group in SPECTRAL_GROUPS
    )
    return RSRRelease(
        payload, serialized, reconstructed, mask_rows, frozenset(support),
        total_spectral, released_spectral, spectral_entry_bytes,
        hashlib.sha256(serialized).hexdigest(), tensor_mapping_sha256(reconstructed),
    )


def mask_seed_policy() -> dict[str, Any]:
    return {
        "version": MASK_POLICY_VERSION, "namespace_hex": MASK_NAMESPACE.hex(),
        "hash": "SHA-256",
        "integer_derivation": "first_8_bytes_big_endian_masked_to_63_bits",
        "fields": ["experiment_seed", "client_id", "round_id", "release_index",
                   "group_name", "tensor_name"],
        "python_builtin_hash_used": False, "secret_randomness_used": False,
        "mask_public_to_server": True, "data_dependent": False,
        "magnitude_dependent": False, "property_dependent": False,
    }
