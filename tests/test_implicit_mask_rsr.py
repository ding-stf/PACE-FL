from __future__ import annotations
import pytest

import csv
import inspect
import json
from pathlib import Path

import torch

from communication.implicit_mask_rsr import (
    IMPLICIT_MAGIC,
    PublicMaskContext,
    decode_implicit_mask_payload,
    encode_implicit_mask_payload,
    logical_payload_sha256,
    payloads_exact,
    schema_from_state_dict,
    transmitted_value_bytes,
)
from communication.randomized_spectral_release import (
    RSREntry,
    RSRPayload,
    SPECTRAL_GROUPS,
    deserialize_rsr_payload,
    randomized_spectral_release,
    reconstruct_rsr_update,
    tensor_mapping_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/randomized_spectral_release_sanity/pems08/seed42/p050"
AUDIT = ROOT / "results/randomized_spectral_release_sanity/communication_audit/implicit_mask"


def sample_update(scale: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        "graph_conv.theta": torch.arange(48, dtype=torch.float32).reshape(6, 4, 2) * scale,
        "temporal_in.weight": torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5) * scale,
        "temporal_out.weight": torch.arange(120, 240, dtype=torch.float32).reshape(2, 3, 4, 5) * scale,
        "temporal_in.bias": torch.arange(6, dtype=torch.float32) * scale,
    }


def synthetic():
    fields = dict(p=0.5, experiment_seed=42, client_id=7, round_id=3, release_index=0)
    release = randomized_spectral_release(sample_update(), **fields)
    schema = schema_from_state_dict(sample_update())
    context = PublicMaskContext(42, 7, 3, 0, 0.5)
    return release, schema, context


def archived_contexts():
    with (ARCHIVE / "communication_stats.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["Round"]), int(row["ClientID"])): PublicMaskContext(
            42, int(row["ClientID"]), int(row["Round"]), int(row["ReleaseIndex"]), 0.5
        )
        for row in rows
    }


def archived_schema():
    state = torch.load(ARCHIVE / "final_model.pt", map_location="cpu", weights_only=False)
    return schema_from_state_dict(state)


def load_layers(path: Path):
    record = torch.load(path, map_location="cpu", weights_only=False)
    return record.layers if hasattr(record, "layers") else record


def test_implicit_encoder_decoder_round_trip():
    release, schema, context = synthetic()
    encoded = encode_implicit_mask_payload(release.payload, schema, context)
    decoded = decode_implicit_mask_payload(encoded, schema, context)
    assert payloads_exact(release.payload, decoded)
    assert logical_payload_sha256(release.payload) == logical_payload_sha256(decoded)


def test_no_indices_in_wire_format():
    release, schema, context = synthetic()
    encoded = encode_implicit_mask_payload(release.payload, schema, context)
    assert encoded[:4] == IMPLICIT_MAGIC
    assert len(encoded) == 4 + transmitted_value_bytes(release.payload)
    explicit_index_bytes = sum(
        entry.indices.numel() * entry.indices.element_size()
        for entry in release.payload.entries if entry.indices is not None
    )
    assert explicit_index_bytes > 0


def test_deterministic_mask_regeneration():
    release, schema, context = synthetic()
    encoded = encode_implicit_mask_payload(release.payload, schema, context)
    left = decode_implicit_mask_payload(encoded, schema, context)
    right = decode_implicit_mask_payload(encoded, schema, context)
    for a, b in zip(left.entries, right.entries):
        if a.indices is not None:
            assert torch.equal(a.indices, b.indices)


def test_implicit_mask_blind_decode():
    release, schema, context = synthetic()
    expected_hash = logical_payload_sha256(release.payload)
    wire = encode_implicit_mask_payload(release.payload, schema, context)
    del release
    decoded = decode_implicit_mask_payload(wire, schema, context)
    assert logical_payload_sha256(decoded) == expected_hash
    parameters = set(inspect.signature(decode_implicit_mask_payload).parameters)
    assert parameters == {"data", "schema", "context"}


@pytest.mark.skipif(not (ARCHIVE / "final_model.pt").exists(), reason="Optional original experiment archive is not distributed")
def test_100_archived_payload_reconstruction_audit():
    schema = archived_schema()
    contexts = archived_contexts()
    paths = sorted((ARCHIVE / "server_visible_payloads").glob("*.rsr"))
    assert len(paths) == 100
    exact = 0
    for path in paths:
        parts = path.stem.split("_")
        key = (int(parts[1]), int(parts[3]))
        old = deserialize_rsr_payload(path.read_bytes())
        wire = encode_implicit_mask_payload(old, schema, contexts[key])
        new = decode_implicit_mask_payload(wire, schema, contexts[key])
        exact += int(payloads_exact(old, new) and logical_payload_sha256(old) == logical_payload_sha256(new))
    assert exact == 100


@pytest.mark.skipif(not (ARCHIVE / "final_model.pt").exists(), reason="Optional original experiment archive is not distributed")
def test_100_server_visible_reconstructed_updates_equal():
    schema = archived_schema()
    contexts = archived_contexts()
    paths = sorted((ARCHIVE / "server_visible_payloads").glob("*.rsr"))
    exact = 0
    for path in paths:
        parts = path.stem.split("_")
        key = (int(parts[1]), int(parts[3]))
        old = deserialize_rsr_payload(path.read_bytes())
        wire = encode_implicit_mask_payload(old, schema, contexts[key])
        decoded = decode_implicit_mask_payload(wire, schema, contexts[key])
        reconstructed = reconstruct_rsr_update(decoded)
        archived = load_layers(ARCHIVE / "server_visible_updates" / f"{path.stem}.pt")
        exact += int(tensor_mapping_sha256(reconstructed) == tensor_mapping_sha256(archived))
    assert exact == 100


@pytest.mark.skipif(not (AUDIT / "manifest.json").exists(), reason="Optional original experiment archive is not distributed")
def test_communication_byte_accounting_exactness():
    manifest = json.loads((AUDIT / "manifest.json").read_text(encoding="utf-8"))
    files = sorted((AUDIT / "implicit_payloads").glob("*.imr"))
    assert len(files) == 100
    assert sum(path.stat().st_size for path in files) == manifest["implicit_rsr_protocol_bytes"]
    assert manifest["implicit_rsr_protocol_bytes"] == manifest["implicit_raw_value_bytes"] + 400
    assert manifest["dense_protocol_bytes"] == 808100
    assert manifest["explicit_rsr_protocol_bytes"] == 838620


def test_changing_coefficient_values_does_not_alter_mask():
    release, schema, context = synthetic()
    changed_entries = []
    for entry in release.payload.entries:
        changed_entries.append(RSREntry(
            entry.group, entry.name, entry.shape, entry.values + 1.0,
            None if entry.indices is None else entry.indices.clone(),
        ))
    changed = RSRPayload(release.payload.p, tuple(changed_entries))
    original_decoded = decode_implicit_mask_payload(
        encode_implicit_mask_payload(release.payload, schema, context), schema, context
    )
    changed_decoded = decode_implicit_mask_payload(
        encode_implicit_mask_payload(changed, schema, context), schema, context
    )
    for old, new in zip(original_decoded.entries, changed_decoded.entries):
        if old.indices is not None:
            assert torch.equal(old.indices, new.indices)
        assert not torch.equal(old.values, new.values)


def test_different_public_context_changes_mask():
    first = randomized_spectral_release(
        sample_update(), p=0.5, experiment_seed=42, client_id=7, round_id=3, release_index=0
    )
    second = randomized_spectral_release(
        sample_update(), p=0.5, experiment_seed=42, client_id=8, round_id=4, release_index=0
    )
    first_masks = {
        (e.group, e.name): e.indices for e in first.payload.entries if e.group in SPECTRAL_GROUPS
    }
    second_masks = {
        (e.group, e.name): e.indices for e in second.payload.entries if e.group in SPECTRAL_GROUPS
    }
    assert any(not torch.equal(first_masks[key], second_masks[key]) for key in first_masks)


def test_no_quantization_dp_topk_or_error_feedback_introduced():
    source = (ROOT / "communication/implicit_mask_rsr.py").read_text(encoding="utf-8").lower()
    forbidden = [
        "torch.randn", "privacy.clipping", ".topk(", "error_feedback",
        "float16", "int8", "gzip", "zstd", "lz4",
    ]
    assert all(token not in source for token in forbidden)


def test_other_parameters_remain_dense_and_exact():
    release, schema, context = synthetic()
    decoded = decode_implicit_mask_payload(
        encode_implicit_mask_payload(release.payload, schema, context), schema, context
    )
    old = {(e.group, e.name): e for e in release.payload.entries}
    for entry in decoded.entries:
        if entry.group == "other":
            assert entry.indices is None
            assert torch.equal(entry.values, old[(entry.group, entry.name)].values)


def test_public_schema_canonical_order_is_fixed():
    _, schema, _ = synthetic()
    keys = [(entry.group, entry.name) for entry in schema.entries]
    assert keys == sorted(keys)
