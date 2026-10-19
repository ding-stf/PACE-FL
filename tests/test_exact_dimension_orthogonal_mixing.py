from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from communication.exact_dimension_orthogonal_mixing import (
    GLOBAL_MIX_SEED,
    TransformContext,
    deserialize_ed_payload,
    exact_dimension_rsr_release,
    exact_orthogonal_inverse,
    exact_orthogonal_mix,
    group_dimensions,
    serialize_ed_payload,
    stage_count,
    stage_permutations,
)
from communication.implicit_mask_rsr import PublicMaskContext, schema_from_state_dict
from communication.randomized_spectral_release import deserialize_rsr_payload, tensor_mapping_sha256

ROOT = Path(__file__).resolve().parents[1]
RSR = ROOT / "results/randomized_spectral_release_sanity/pems08/seed42/p050"
IMR = ROOT / "results/randomized_spectral_release_sanity/communication_audit/implicit_mask"


def sample_update(seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return {
        "graph_conv.theta": torch.randn(3, 16, 16, generator=g),
        "graph_conv.bias": torch.randn(16, generator=g),
        "temporal_in.weight": torch.randn(16, 3, 3, 1, generator=g),
        "temporal_in.bias": torch.randn(16, generator=g),
        "temporal_out.weight": torch.randn(16, 16, 3, 1, generator=g),
        "temporal_out.bias": torch.randn(16, generator=g),
        "readout.weight": torch.randn(12, 16, generator=g),
        "readout.bias": torch.randn(12, generator=g),
    }


@pytest.mark.parametrize("d", [1, 2, 3, 5, 7, 10, 17, 63, 100, 304, 608])
def test_t1_dimension_preservation(d):
    x = torch.randn(d, generator=torch.Generator().manual_seed(d), dtype=torch.float64)
    context = TransformContext(GLOBAL_MIX_SEED, 2, 1, "temporal_low")
    y, _ = exact_orthogonal_mix(x, context)
    assert y.shape == x.shape
    assert stage_count(d) == int(np.ceil(np.log2(max(d, 2))))


def test_t2_locked_schema_has_no_padding():
    schema = schema_from_state_dict(sample_update())
    dims = group_dimensions(schema)
    assert dims == {"graph_low": 512, "graph_high": 256, "temporal_low": 608, "temporal_high": 304}
    release = exact_dimension_rsr_release(sample_update(), p=.5, experiment_seed=42, client_id=4, round_id=2, release_index=0, schema=schema)
    assert release.original_coordinates == release.mixed_coordinates == 1680
    assert all(row["ExtraMaskCoordinates"] == 0 for row in release.mask_rows)


@pytest.mark.parametrize("d", [1, 2, 3, 5, 10, 17, 63, 100, 608])
def test_t3_orthogonal_inverse(d):
    x = torch.randn(d, generator=torch.Generator().manual_seed(100 + d), dtype=torch.float64)
    context = TransformContext(GLOBAL_MIX_SEED, 3, 0, "temporal_high")
    y, _ = exact_orthogonal_mix(x, context)
    recovered = exact_orthogonal_inverse(y, context)
    assert torch.max(torch.abs(recovered - x)).item() < 2e-12


@pytest.mark.parametrize("d", [1, 3, 17, 100, 608])
def test_t4_norm_preservation(d):
    x = torch.randn(d, generator=torch.Generator().manual_seed(200 + d), dtype=torch.float64)
    context = TransformContext(GLOBAL_MIX_SEED, 7, 1, "graph_low")
    y, _ = exact_orthogonal_mix(x, context)
    assert abs(float(torch.linalg.vector_norm(x)) - float(torch.linalg.vector_norm(y))) < 3e-12


def test_t5_p1_exact_server_reconstruction():
    update = sample_update()
    release = exact_dimension_rsr_release(update, p=1.0, experiment_seed=42, client_id=8, round_id=6, release_index=1)
    for name in update:
        assert torch.allclose(release.reconstructed_update[name], update[name], rtol=3e-5, atol=3e-6), name


@pytest.mark.skipif(not (RSR / "server_visible_update_hashes.json").exists(), reason="Optional original experiment archive is not distributed")
def test_t6_t7_all_100_masks_counts_and_imr_lengths_exact():
    hash_doc = json.loads((RSR / "server_visible_update_hashes.json").read_text())
    releases = {(int(r["Round"]), int(r["ClientID"])): int(r["ReleaseIndex"]) for r in hash_doc["records"]}
    record = torch.load(sorted((RSR / "server_visible_updates").glob("*.pt"))[0], map_location="cpu", weights_only=False)
    zero_update = {name: torch.zeros_like(value) for name, value in record.layers.items()}
    schema = schema_from_state_dict(zero_update)
    passed = 0
    for path in sorted((RSR / "server_visible_payloads").glob("*.rsr")):
        part = path.stem.split("_"); round_id, client_id = int(part[1]), int(part[3])
        release_index = releases[(round_id, client_id)]
        canonical = deserialize_rsr_payload(path.read_bytes())
        ed = exact_dimension_rsr_release(zero_update, p=.5, experiment_seed=42, client_id=client_id, round_id=round_id, release_index=release_index, schema=schema)
        expected = {(e.group, e.name): e for e in canonical.entries}
        actual = {(e.group, e.name): e for e in ed.payload.entries}
        assert set(actual) == set(expected)
        for key in expected:
            if expected[key].indices is None:
                assert actual[key].indices is None
            else:
                assert torch.equal(actual[key].indices, expected[key].indices), (round_id, client_id, key)
                assert actual[key].values.numel() == expected[key].values.numel()
        imr_path = IMR / "implicit_payloads" / path.name.replace(".rsr", ".imr")
        assert len(ed.serialized) == imr_path.stat().st_size
        passed += 1
    assert passed == 100


def test_t8_transform_is_client_independent_by_api_and_metadata():
    x = torch.randn(17, generator=torch.Generator().manual_seed(9))
    context = TransformContext(GLOBAL_MIX_SEED, 4, 1, "graph_low")
    a, ma = exact_orthogonal_mix(x, context); b, mb = exact_orthogonal_mix(x, context)
    assert torch.equal(a, b) and ma == mb
    assert "client" not in json.dumps(ma).lower().replace("clientidincluded", "")
    assert ma["ClientIDIncluded"] is False


def test_t9_input_value_independence():
    context = TransformContext(GLOBAL_MIX_SEED, 9, 0, "temporal_low")
    _, first = exact_orthogonal_mix(torch.zeros(31), context)
    _, second = exact_orthogonal_mix(torch.arange(31, dtype=torch.float32), context)
    assert first == second


@pytest.mark.parametrize("d", [3, 5, 7, 9, 17, 63])
def test_t10_odd_dimension_leftover_fairness(d):
    context = TransformContext(GLOBAL_MIX_SEED, 12, 1, "temporal_high")
    _, stages = stage_permutations(d, context)
    leftovers = [row.leftover_input_slot for row in stages]
    assert len(set(leftovers)) > 1
    assert all(a != b for a, b in zip(leftovers, leftovers[1:]))


def test_t11_server_reconstruction_is_payload_and_public_context_only():
    raw = sample_update(); schema = schema_from_state_dict(raw)
    release = exact_dimension_rsr_release(raw, p=.5, experiment_seed=42, client_id=6, round_id=4, release_index=0, schema=schema)
    data = bytes(release.serialized); del raw
    public = PublicMaskContext(42, 6, 4, 0, .5)
    decoded = deserialize_ed_payload(data, schema, public)
    from communication.exact_dimension_orthogonal_mixing import reconstruct_ed_update
    update, _ = reconstruct_ed_update(decoded, schema, experiment_seed=42, client_id=6, round_id=4, release_index=0)
    assert tensor_mapping_sha256(update) == release.reconstructed_sha256


def test_t12_serialization_roundtrip_hash_and_no_transform_metadata():
    raw = sample_update(); schema = schema_from_state_dict(raw)
    first = exact_dimension_rsr_release(raw, p=.5, experiment_seed=42, client_id=6, round_id=4, release_index=0, schema=schema)
    second = exact_dimension_rsr_release(raw, p=.5, experiment_seed=42, client_id=6, round_id=4, release_index=0, schema=schema)
    context = PublicMaskContext(42, 6, 4, 0, .5)
    decoded = deserialize_ed_payload(first.serialized, schema, context)
    assert serialize_ed_payload(decoded, schema, context) == first.serialized == second.serialized
    assert hashlib.sha256(first.serialized).hexdigest() == first.payload_sha256 == second.payload_sha256
    assert first.payload_metadata_bytes == 4
    for forbidden in (b"SignSeed", b"Permutation", b"graph_low", b"temporal_low"):
        assert forbidden not in first.serialized
