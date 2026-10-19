from __future__ import annotations

"""Experiment 18 frozen attacks, seed-level aggregation, tests, and report."""

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import yaml

from attacks.ed_om_rsr_attack import (
    PublicReleaseContext,
    feature_provenance as ed_feature_provenance,
    full_transform_aware_features,
    release_value_features as ed_release_value_features,
)
from attacks.mask_aware_spectral_attack import (
    adaptive_feature_provenance,
    build_mask_aware_feature_vector,
)
from attacks.rsr_leakage_decomposition import release_value_features as rsr_release_value_features
from attacks.spectral_leakage.attack_dataset import raw_properties
from communication.exact_dimension_orthogonal_mixing import deserialize_ed_payload
from communication.implicit_mask_rsr import (
    PublicMaskContext,
    PublicRSRSchema,
    decode_implicit_mask_payload,
)
from communication.randomized_spectral_release import deserialize_rsr_payload
from datasets.client_partition import partition_nodes
from datasets.stg_utils import temporal_split
from datasets.traffic import load_traffic_dataset
from experiments import exp_mask_aware_adaptive_attack as e13
from experiments import exp_randomized_spectral_release_sanity as e11
from experiments.common import write_rows
from experiments.ed_om_rsr_multiseed_common import (
    communication_preserved,
    excess_reduction,
    payload_pair_audit,
    replication_judgment,
    sample_mean_std,
    validate_seed_method_rows,
)
from experiments.exp_representative_privacy_attack import extract_feature_families, make_folds

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/ed_om_rsr_multiseed/pems08"
SUMMARY = BASE / "summary"
REPORT = ROOT / "docs/ED_OM_RSR_MULTISEED_REPORT.md"
CANON42 = ROOT / "results/randomized_spectral_release_sanity/pems08/seed42/p050"
IMR42 = ROOT / "results/randomized_spectral_release_sanity/communication_audit/implicit_mask"
E17 = ROOT / "results/ed_om_rsr_sanity/pems08/seed42/p050"
RUN_MANIFEST = ROOT / "docs/ED_OM_RSR_MULTISEED_RUN_MANIFEST.json"
RUN_MANIFEST_HASH = ROOT / "docs/ED_OM_RSR_MULTISEED_RUN_MANIFEST.sha256"
PROPERTY_SPEC = ROOT / "docs/R2_ATTACK_FEATURE_SPEC.yaml"
PROTOCOL_SPEC = ROOT / "docs/ED_OM_RSR_PROTOCOL_SPEC.md"
ATTACK_SPEC = ROOT / "docs/ED_OM_RSR_ATTACK_FEATURE_SPEC.md"
PRETEST = BASE / ".exp18_pretests.txt"
RUN_MANIFEST_SHA256 = "448497e56492b8fb63d3dcedc57646f87460ba67df7bb03781a13ce4c5ab9fdb"
PROTOCOL_SHA256 = "8de7512141e019201005fcc5656b9fe8cec908e8735161da56c1a5753fdfe7ae"
ATTACK_SHA256 = "99ac253ebaaafd43e035a627d7a5ba2415904488a11a49e4c2148cf70a7f013a"
PROPERTY_SPEC_SHA256 = "260e57f6c99c94dde01bb3b09eb27bbc888c39c70c69a454d50d1dd3dec6af7d"
SEEDS = (42, 123, 2026)
PROPERTIES = ("Extreme Event", "Periodicity Strength")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return e11.canonical_hash(value)


def metric(table: list[dict[str, Any]], prop: str, condition: str) -> dict[str, Any]:
    matches = [
        row for row in table
        if row["Property"] == prop and row["AttackCondition"] == condition
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one metric for {prop}/{condition}, got {len(matches)}")
    return matches[0]


def verify_output_hashes(run_dir: Path, manifest: dict[str, Any]) -> None:
    bad = [
        name for name, expected in manifest["output_hashes"].items()
        if not (run_dir / name).is_file() or sha(run_dir / name) != expected
    ]
    if bad:
        raise RuntimeError(f"paired run output hash failure: {run_dir}: {bad}")


def input_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    required = [
        RUN_MANIFEST,
        RUN_MANIFEST_HASH,
        PROPERTY_SPEC,
        PROTOCOL_SPEC,
        ATTACK_SPEC,
        PRETEST,
        E17 / "manifest.json",
        E17 / "attack_metrics.csv",
        E17 / "utility_metrics.json",
        E17 / "communication_metrics.csv",
        CANON42 / "manifest.json",
        CANON42 / "metrics.csv",
        CANON42 / "participation_schedule.json",
        IMR42 / "payload_audit.csv",
    ]
    for seed in (123, 2026):
        for method in ("rsr", "ed_om_rsr"):
            required.extend([
                BASE / f"seed{seed}" / method / "manifest.json",
                BASE / f"seed{seed}" / method / "utility_metrics.json",
                BASE / f"seed{seed}" / method / "communication_metrics.csv",
            ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment 18 formal input missing: {missing}")
    if (
        sha(RUN_MANIFEST) != RUN_MANIFEST_SHA256
        or RUN_MANIFEST_HASH.read_text(encoding="utf-8").split()[0] != RUN_MANIFEST_SHA256
        or sha(PROTOCOL_SPEC) != PROTOCOL_SHA256
        or sha(ATTACK_SPEC) != ATTACK_SHA256
        or sha(PROPERTY_SPEC) != PROPERTY_SPEC_SHA256
    ):
        raise RuntimeError("Experiment 18 frozen protocol/run/attack input changed")
    passed, failed = e11.parse_tests(PRETEST.read_text(encoding="utf-8"))
    if failed or passed < 59:
        raise RuntimeError("Experiment 18 pre-training tests did not pass")
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    e17 = json.loads((E17 / "manifest.json").read_text(encoding="utf-8"))
    if (
        e17["judgment"] != "SUPPORTED"
        or e17["proceed_to_ed_om_rsr_multiseed"] != "YES"
        or sha(ROOT / "docs/ED_OM_RSR_SANITY_REPORT.md") != e17["report_sha256"]
    ):
        raise RuntimeError("EXPERIMENT 18 BLOCKED: EXPERIMENT 17 REFERENCE MISMATCH")
    e17_metrics = rows(E17 / "attack_metrics.csv")
    checks = {
        "periodicity_rsr_full": float(metric(e17_metrics, "Periodicity Strength", "RSR Full Mask-Aware")["ROCAUCSeparability"]),
        "periodicity_ed_full": float(metric(e17_metrics, "Periodicity Strength", "ED-OM-RSR Full Transform-Aware")["ROCAUCSeparability"]),
        "periodicity_rsr_r1": float(metric(e17_metrics, "Periodicity Strength", "RSR Release-1 Values")["ROCAUCSeparability"]),
        "periodicity_ed_r1": float(metric(e17_metrics, "Periodicity Strength", "ED-OM-RSR Release-1 Values")["ROCAUCSeparability"]),
        "extreme_ed_full": float(metric(e17_metrics, "Extreme Event", "ED-OM-RSR Full Transform-Aware")["ROCAUCSeparability"]),
    }
    calculated_reduction = excess_reduction(checks["periodicity_rsr_full"], checks["periodicity_ed_full"])
    if (
        calculated_reduction is None
        or not np.isclose(calculated_reduction, e17["privacy"]["periodicity_excess_reduction_vs_rsr"])
        or not np.isclose(checks["extreme_ed_full"], e17["privacy"]["extreme_ed_om_full_transform_aware"])
        or int(e17["communication"]["EDOMRSRBytes"]) != 437460
    ):
        raise RuntimeError("EXPERIMENT 18 BLOCKED: EXPERIMENT 17 REFERENCE MISMATCH")
    return run_manifest, {"passed": passed, "failed": failed}


def labels_and_folds(seed: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]], str, str]:
    spec = yaml.safe_load(PROPERTY_SPEC.read_text(encoding="utf-8"))
    data = load_traffic_dataset(ROOT / "data", "pems08")
    private_train = temporal_split(data.values).train
    partition = partition_nodes(data.adjacency, 50, data.coordinates, int(seed))
    properties = raw_properties([private_train[:, nodes] for nodes in partition.node_ids])
    score_keys = {
        "Extreme Event": "extreme_score",
        "Periodicity Strength": "periodicity_score",
    }
    spec_keys = {
        "Extreme Event": "extreme_event",
        "Periodicity Strength": "periodicity_strength",
    }
    result = {}
    records = []
    fold_only = []
    for prop in PROPERTIES:
        property_spec = spec["properties"][spec_keys[prop]]
        scores = np.asarray(properties[score_keys[prop]], dtype=np.float64)
        threshold = float(property_spec["binary_threshold"])
        labels = (scores >= threshold).astype(int)
        minority = int(min(labels.sum(), len(labels) - labels.sum()))
        n_splits = min(5, minority)
        if n_splits < 3:
            raise RuntimeError(f"insufficient property support for seed{seed}/{prop}")
        folds = make_folds(labels, n_splits)
        result[prop] = (labels, folds)
        for client_id in range(50):
            row = {
                "Seed": seed,
                "Property": prop,
                "ClientID": client_id,
                "Score": float(scores[client_id]),
                "Threshold": threshold,
                "Label": int(labels[client_id]),
                "Fold": int(folds[client_id]),
                "NumFolds": n_splits,
            }
            records.append(row)
            fold_only.append({
                "Seed": seed,
                "Property": prop,
                "ClientID": client_id,
                "Fold": int(folds[client_id]),
            })
    return result, records, canonical_hash(records), canonical_hash(fold_only)


def load_new_pair(seed: int):
    rsr_dir = BASE / f"seed{seed}/rsr"
    ed_dir = BASE / f"seed{seed}/ed_om_rsr"
    rsr_manifest = json.loads((rsr_dir / "manifest.json").read_text(encoding="utf-8"))
    ed_manifest = json.loads((ed_dir / "manifest.json").read_text(encoding="utf-8"))
    for run_dir, manifest in ((rsr_dir, rsr_manifest), (ed_dir, ed_manifest)):
        verify_output_hashes(run_dir, manifest)
        if (
            manifest["seed"] != seed
            or manifest["release_count"] != 100
            or manifest["run_manifest_sha256"] != RUN_MANIFEST_SHA256
            or manifest["protocol_spec_sha256"] != PROTOCOL_SHA256
            or manifest["attack_spec_sha256"] != ATTACK_SHA256
            or manifest["DP"]
            or manifest["padding"]
        ):
            raise RuntimeError(f"paired run provenance lock failed: {run_dir}")
    if rsr_manifest["method"] != "RSR" or ed_manifest["method"] != "ED-OM-RSR":
        raise RuntimeError("paired method identities failed")
    if rsr_manifest["schedule_sha256"] != ed_manifest["schedule_sha256"]:
        raise RuntimeError("paired schedule differs")
    if rsr_manifest["public_schema_sha256"] != ed_manifest["public_schema_sha256"]:
        raise RuntimeError("paired public schema differs")

    schema = PublicRSRSchema.from_dict(json.loads((rsr_dir / "public_schema.json").read_text(encoding="utf-8")))
    rsr_mapping = {
        (int(row["Round"]), int(row["ClientID"])): row
        for row in rows(rsr_dir / "round_release_mapping.csv")
    }
    ed_mapping = {
        (int(row["Round"]), int(row["ClientID"])): row
        for row in rows(ed_dir / "round_release_mapping.csv")
    }
    if set(rsr_mapping) != set(ed_mapping) or len(rsr_mapping) != 100:
        raise RuntimeError("paired release mapping differs")

    rsr_payloads: defaultdict[int, list] = defaultdict(list)
    ed_payloads: defaultdict[int, list] = defaultdict(list)
    rsr_records: defaultdict[int, list] = defaultdict(list)
    ed_records: defaultdict[int, list] = defaultdict(list)
    pair_rows = []
    for key in sorted(rsr_mapping):
        rsr_row, ed_row = rsr_mapping[key], ed_mapping[key]
        if int(rsr_row["ReleaseIndex"]) != int(ed_row["ReleaseIndex"]):
            raise RuntimeError("paired release index differs")
        round_id, client_id = key
        release_index = int(rsr_row["ReleaseIndex"])
        context = PublicMaskContext(seed, client_id, round_id, release_index, 0.5)
        public_context = PublicReleaseContext(client_id, round_id, release_index)
        rsr_path = rsr_dir / "server_visible_payloads" / rsr_row["PayloadFile"]
        ed_path = ed_dir / "server_visible_payloads" / ed_row["PayloadFile"]
        rsr_payload = decode_implicit_mask_payload(rsr_path.read_bytes(), schema, context)
        ed_payload = deserialize_ed_payload(ed_path.read_bytes(), schema, context)
        audit = payload_pair_audit(rsr_payload, ed_payload)
        pair_rows.append({
            "Seed": seed,
            "Round": round_id,
            "ClientID": client_id,
            "ReleaseIndex": release_index,
            **audit,
            "RSRSerializedBytes": rsr_path.stat().st_size,
            "EDOMSerializedBytes": ed_path.stat().st_size,
            "SerializedBytesExact": rsr_path.stat().st_size == ed_path.stat().st_size,
        })
        rsr_payloads[client_id].append((round_id, rsr_payload))
        ed_payloads[client_id].append((public_context, ed_payload))
        rsr_update = rsr_dir / "server_visible_updates" / rsr_path.with_suffix(".pt").name
        ed_update = ed_dir / "server_visible_updates" / ed_path.with_suffix(".pt").name
        rsr_records[client_id].append(torch.load(rsr_update, map_location="cpu", weights_only=False))
        ed_records[client_id].append(torch.load(ed_update, map_location="cpu", weights_only=False))
    if not all(
        row["SchemaMatch"]
        and row["MaskMatch"]
        and row["RetainedCountMatch"]
        and row["SerializedBytesExact"]
        for row in pair_rows
    ):
        raise RuntimeError(f"seed{seed} paired payload gate failed")
    for client_id in range(50):
        rsr_payloads[client_id].sort(key=lambda item: item[0])
        ed_payloads[client_id].sort(key=lambda item: item[0].round_id)
        rsr_records[client_id].sort(key=lambda item: item.round)
        ed_records[client_id].sort(key=lambda item: item.round)
    return {
        "rsr_dir": rsr_dir,
        "ed_dir": ed_dir,
        "rsr_manifest": rsr_manifest,
        "ed_manifest": ed_manifest,
        "schema": schema,
        "rsr_payloads": rsr_payloads,
        "ed_payloads": ed_payloads,
        "rsr_records": rsr_records,
        "ed_records": ed_records,
        "pair_rows": pair_rows,
    }


def evaluate_new_seed(seed: int, pair: dict[str, Any], labels_folds: dict[str, tuple[np.ndarray, np.ndarray]]):
    rsr_features: dict[str, list[np.ndarray]] = {
        "RSR Standard": [],
        "RSR Full Adaptive": [],
        "RSR Release-1 Values": [],
        "RSR Release-2 Values": [],
    }
    ed_features: dict[str, list[np.ndarray]] = {
        "ED-OM-RSR Standard": [],
        "ED-OM-RSR Full Transform-Aware": [],
        "ED-OM-RSR Release-1 Values": [],
        "ED-OM-RSR Release-2 Values": [],
    }
    schemas: dict[str, tuple[str, ...]] = {}
    feature_rows = []
    provenance_rows = []
    for client_id in range(50):
        rsr_family = extract_feature_families([
            e11.attack_record(record) for record in pair["rsr_records"][client_id]
        ])
        ed_family = extract_feature_families([
            e11.attack_record(record) for record in pair["ed_records"][client_id]
        ])
        rsr_pair = [payload for _, payload in pair["rsr_payloads"][client_id]]
        ed_contexts = tuple(context for context, _ in pair["ed_payloads"][client_id])
        ed_pair = tuple(payload for _, payload in pair["ed_payloads"][client_id])
        rsr_vectors = {
            "RSR Standard": (
                tuple(f"standard__{index:03d}" for index in range(24)),
                rsr_family["proposed_spectral"],
            ),
            "RSR Full Adaptive": (
                build_mask_aware_feature_vector(rsr_pair, rsr_family["proposed_spectral"]).names,
                build_mask_aware_feature_vector(rsr_pair, rsr_family["proposed_spectral"]).values,
            ),
            "RSR Release-1 Values": (
                rsr_release_value_features(rsr_pair, 1).names,
                rsr_release_value_features(rsr_pair, 1).values,
            ),
            "RSR Release-2 Values": (
                rsr_release_value_features(rsr_pair, 2).names,
                rsr_release_value_features(rsr_pair, 2).values,
            ),
        }
        ed_full = full_transform_aware_features(
            ed_family["proposed_spectral"],
            ed_pair,
            ed_contexts,
            pair["schema"],
        )
        ed_r1 = ed_release_value_features(ed_pair, ed_contexts, pair["schema"], 1)
        ed_r2 = ed_release_value_features(ed_pair, ed_contexts, pair["schema"], 2)
        ed_vectors = {
            "ED-OM-RSR Standard": (
                tuple(f"standard__{index:03d}" for index in range(24)),
                ed_family["proposed_spectral"],
            ),
            "ED-OM-RSR Full Transform-Aware": (ed_full.names, ed_full.values),
            "ED-OM-RSR Release-1 Values": (ed_r1.names, ed_r1.values),
            "ED-OM-RSR Release-2 Values": (ed_r2.names, ed_r2.values),
        }
        for condition, (names, values) in {**rsr_vectors, **ed_vectors}.items():
            names = tuple(names)
            values = np.asarray(values, dtype=np.float64)
            if condition in schemas and schemas[condition] != names:
                raise RuntimeError(f"seed{seed} attack schema changed: {condition}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"seed{seed} non-finite attack feature: {condition}")
            schemas[condition] = names
            target = rsr_features if condition.startswith("RSR ") else ed_features
            target[condition].append(values)
            feature_rows.extend({
                "Seed": seed,
                "Condition": condition,
                "ClientID": client_id,
                "FeatureName": name,
                "FeatureValue": float(value),
            } for name, value in zip(names, values))
    matrices = {
        condition: np.stack(values)
        for condition, values in {**rsr_features, **ed_features}.items()
    }
    expected_shapes = {
        "RSR Standard": (50, 24),
        "RSR Full Adaptive": (50, 104),
        "RSR Release-1 Values": (50, 20),
        "RSR Release-2 Values": (50, 20),
        "ED-OM-RSR Standard": (50, 24),
        "ED-OM-RSR Full Transform-Aware": (50, 168),
        "ED-OM-RSR Release-1 Values": (50, 28),
        "ED-OM-RSR Release-2 Values": (50, 28),
    }
    if any(matrices[name].shape != shape for name, shape in expected_shapes.items()):
        raise RuntimeError(f"seed{seed} frozen attack dimension changed")
    rsr_provenance = adaptive_feature_provenance(schemas["RSR Full Adaptive"])
    ed_provenance = ed_feature_provenance(schemas["ED-OM-RSR Full Transform-Aware"])
    for row in rsr_provenance:
        provenance_rows.append({"Seed": seed, "Method": "RSR", "Condition": "RSR Full Adaptive", **row})
    for row in ed_provenance:
        provenance_rows.append({"Seed": seed, "Method": "ED-OM-RSR", "Condition": "ED-OM-RSR Full Transform-Aware", **row})

    attack_metrics = []
    bootstrap_rows = []
    prediction_rows = []
    for prop in PROPERTIES:
        labels, folds = labels_folds[prop]
        for condition, matrix in matrices.items():
            method_name = "RSR" if condition.startswith("RSR ") else "ED-OM-RSR"
            row, bootstrap, predictions, _ = e13.evaluate(
                prop,
                condition,
                labels,
                folds,
                x=matrix,
                origin=f"Experiment 18 frozen attack, seed{seed}",
            )
            row.update({"Seed": seed, "Method": method_name})
            bootstrap.update({"Seed": seed, "Method": method_name})
            for prediction in predictions:
                prediction.update({"Seed": seed, "Method": method_name})
            attack_metrics.append(row)
            bootstrap_rows.append(bootstrap)
            prediction_rows.extend(predictions)
    return {
        "metrics": attack_metrics,
        "bootstrap": bootstrap_rows,
        "predictions": prediction_rows,
        "features": feature_rows,
        "provenance": provenance_rows,
        "schemas": schemas,
    }


def seed42_reference():
    e17_manifest = json.loads((E17 / "manifest.json").read_text(encoding="utf-8"))
    e17_metrics = rows(E17 / "attack_metrics.csv")
    schema = PublicRSRSchema.from_dict(
        json.loads((E17 / "mixing_provenance.json").read_text(encoding="utf-8"))["schema"]
    )
    ed_mapping = {
        (int(row["Round"]), int(row["ClientID"])): row
        for row in rows(E17 / "round_release_mapping.csv")
    }
    canonical_hashes = json.loads((CANON42 / "server_visible_update_hashes.json").read_text(encoding="utf-8"))
    canonical_mapping = {
        (int(row["Round"]), int(row["ClientID"])): row
        for row in canonical_hashes["records"]
    }
    imr_bytes = {
        (int(row["round"]), int(row["client"])): int(row["new_implicit_bytes"])
        for row in rows(IMR42 / "payload_audit.csv")
        if row["record_type"] == "payload"
    }
    ed_comm = {
        (int(row["Round"]), int(row["ClientID"])): row
        for row in rows(E17 / "communication_metrics.csv")
    }
    pair_rows = []
    for key in sorted(canonical_mapping):
        round_id, client_id = key
        release_index = int(canonical_mapping[key]["ReleaseIndex"])
        rsr_path = CANON42 / "server_visible_payloads" / f"round_{round_id:04d}_client_{client_id:03d}.rsr"
        ed_path = E17 / "server_visible_payloads" / ed_mapping[key]["PayloadFile"]
        rsr_payload = deserialize_rsr_payload(rsr_path.read_bytes())
        context = PublicMaskContext(42, client_id, round_id, release_index, 0.5)
        ed_payload = deserialize_ed_payload(ed_path.read_bytes(), schema, context)
        audit = payload_pair_audit(rsr_payload, ed_payload)
        pair_rows.append({
            "Seed": 42,
            "Round": round_id,
            "ClientID": client_id,
            "ReleaseIndex": release_index,
            **audit,
            "RSRSerializedBytes": imr_bytes[key],
            "EDOMSerializedBytes": int(ed_comm[key]["TotalSerializedBytes"]),
            "SerializedBytesExact": imr_bytes[key] == int(ed_comm[key]["TotalSerializedBytes"]),
        })
    if len(pair_rows) != 100 or not all(
        row["MaskMatch"] and row["RetainedCountMatch"] and row["SerializedBytesExact"]
        for row in pair_rows
    ):
        raise RuntimeError("seed42 formal pairing audit failed")
    labels_table = rows(CANON42 / "attack/property_labels.csv")
    fold_files = [
        CANON42 / "attack/fold_assignments/Extreme_Event_folds.csv",
        CANON42 / "attack/fold_assignments/Periodicity_Strength_folds.csv",
    ]
    label_hash = sha(CANON42 / "attack/property_labels.csv")
    fold_hash = canonical_hash([
        {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
        for path in fold_files
    ])
    utility_rsr = rows(CANON42 / "metrics.csv")[0]
    utility_ed = json.loads((E17 / "utility_metrics.json").read_text(encoding="utf-8"))
    communication = rows(E17 / "communication_metrics.csv")
    comm_rsr = {
        "Seed": 42,
        "Method": "RSR",
        "UploadCount": 100,
        "TotalSerializedBytes": sum(imr_bytes.values()),
        "PayloadBytes": sum(imr_bytes.values()) - 400,
        "MetadataBytes": 400,
        "DenseBytes": sum(int(row["DenseReferenceBytes"]) for row in communication),
    }
    comm_ed = {
        "Seed": 42,
        "Method": "ED-OM-RSR",
        "UploadCount": 100,
        "TotalSerializedBytes": sum(int(row["TotalSerializedBytes"]) for row in communication),
        "PayloadBytes": sum(int(row["ValueBytes"]) for row in communication),
        "MetadataBytes": sum(int(row["MetadataBytes"]) for row in communication),
        "DenseBytes": sum(int(row["DenseReferenceBytes"]) for row in communication),
    }
    return {
        "metrics": e17_metrics,
        "pair_rows": pair_rows,
        "label_hash": label_hash,
        "fold_hash": fold_hash,
        "labels_table": labels_table,
        "utility": [
            {"Seed": 42, "Method": "RSR", **{key: float(utility_rsr[key]) for key in ("MAE", "RMSE", "MAPE")}},
            {"Seed": 42, "Method": "ED-OM-RSR", **{key: float(utility_ed[key]) for key in ("MAE", "RMSE", "MAPE")}},
        ],
        "communication": [comm_rsr, comm_ed],
        "manifest": e17_manifest,
    }


def privacy_rows_for_seed(seed: int, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method_name in ("RSR", "ED-OM-RSR"):
        if method_name == "RSR":
            conditions = {
                "standard": "RSR Standard",
                "full": "RSR Full Mask-Aware" if seed == 42 else "RSR Full Adaptive",
                "r1": "RSR Release-1 Values",
                "r2": "RSR Release-2 Values",
            }
        else:
            conditions = {
                "standard": "ED-OM-RSR Standard",
                "full": "ED-OM-RSR Full Transform-Aware",
                "r1": "ED-OM-RSR Release-1 Values",
                "r2": "ED-OM-RSR Release-2 Values",
            }
        extreme = metric(metrics, "Extreme Event", conditions["full"])
        periodic_standard = metric(metrics, "Periodicity Strength", conditions["standard"])
        periodic_full = metric(metrics, "Periodicity Strength", conditions["full"])
        periodic_r1 = metric(metrics, "Periodicity Strength", conditions["r1"])
        periodic_r2 = metric(metrics, "Periodicity Strength", conditions["r2"])
        result.append({
            "Seed": seed,
            "Method": method_name,
            "ExtremeFullRawAUC": float(extreme["ROCAUCRaw"]),
            "ExtremeFullAUCsep": float(extreme["ROCAUCSeparability"]),
            "ExtremeFullCI95Low": float(extreme["AUCSeparabilityCI95Low"]),
            "ExtremeFullCI95High": float(extreme["AUCSeparabilityCI95High"]),
            "PeriodicityStandardRawAUC": float(periodic_standard["ROCAUCRaw"]),
            "PeriodicityStandardAUCsep": float(periodic_standard["ROCAUCSeparability"]),
            "PeriodicityFullRawAUC": float(periodic_full["ROCAUCRaw"]),
            "PeriodicityFullAUCsep": float(periodic_full["ROCAUCSeparability"]),
            "PeriodicityFullCI95Low": float(periodic_full["AUCSeparabilityCI95Low"]),
            "PeriodicityFullCI95High": float(periodic_full["AUCSeparabilityCI95High"]),
            "PeriodicityRelease1RawAUC": float(periodic_r1["ROCAUCRaw"]),
            "PeriodicityRelease1AUCsep": float(periodic_r1["ROCAUCSeparability"]),
            "PeriodicityRelease2RawAUC": float(periodic_r2["ROCAUCRaw"]),
            "PeriodicityRelease2AUCsep": float(periodic_r2["ROCAUCSeparability"]),
            "EvidenceOrigin": "Experiment 17 formal artifact" if seed == 42 else f"Experiment 18 seed{seed} frozen attack",
        })
    return result


def summarize(rows_in: list[dict[str, Any]], methods: tuple[str, ...], fields: tuple[str, ...]):
    result = []
    for method_name in methods:
        selected = [row for row in rows_in if row["Method"] == method_name]
        if len(selected) != 3:
            raise RuntimeError(f"seed-level summary coverage failed for {method_name}")
        for field in fields:
            mean, std = sample_mean_std([float(row[field]) for row in selected])
            result.append({"Method": method_name, "Metric": field, "Mean": mean, "SampleStd": std, "Seeds": 3, "DDOF": 1})
    return result


def run() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "dsl1":
        raise RuntimeError("Experiment 18 finalization must run in conda environment dsl1")
    if SUMMARY.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite Experiment 18 summary/report")
    run_manifest, pretests = input_audit()
    seed42 = seed42_reference()

    all_attack_artifacts = {}
    label_records_by_seed = {}
    label_hashes = {42: seed42["label_hash"]}
    fold_hashes = {42: seed42["fold_hash"]}
    all_pair_rows = list(seed42["pair_rows"])
    privacy_rows = privacy_rows_for_seed(42, seed42["metrics"])
    utility_rows = list(seed42["utility"])
    communication_rows = list(seed42["communication"])

    for seed in (123, 2026):
        labels_folds, label_records, label_hash, fold_hash = labels_and_folds(seed)
        pair = load_new_pair(seed)
        attack = evaluate_new_seed(seed, pair, labels_folds)
        all_attack_artifacts[seed] = attack
        label_records_by_seed[seed] = label_records
        label_hashes[seed] = label_hash
        fold_hashes[seed] = fold_hash
        all_pair_rows.extend(pair["pair_rows"])
        privacy_rows.extend(privacy_rows_for_seed(seed, attack["metrics"]))
        for method_key, method_name in (("rsr", "RSR"), ("ed_om_rsr", "ED-OM-RSR")):
            run_dir = BASE / f"seed{seed}" / method_key
            utility = json.loads((run_dir / "utility_metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            utility_rows.append({
                "Seed": seed,
                "Method": method_name,
                **{key: float(utility[key]) for key in ("MAE", "RMSE", "MAPE")},
            })
            communication_rows.append({
                "Seed": seed,
                "Method": method_name,
                "UploadCount": manifest["release_count"],
                "TotalSerializedBytes": manifest["serialized_bytes"],
                "PayloadBytes": manifest["payload_value_bytes"],
                "MetadataBytes": manifest["metadata_bytes"],
                "DenseBytes": manifest["dense_reference_bytes"],
            })

    validate_seed_method_rows(privacy_rows)
    validate_seed_method_rows(utility_rows)
    validate_seed_method_rows(communication_rows)
    if len(all_pair_rows) != 300:
        raise RuntimeError("paired release audit must contain 300 rows")

    pair_audit_rows = []
    for seed in SEEDS:
        release_rows = [row for row in all_pair_rows if int(row["Seed"]) == seed]
        if len(release_rows) != 100:
            raise RuntimeError(f"seed{seed} pair coverage failed")
        if seed == 42:
            rsr_schedule = sha(CANON42 / "participation_schedule.json")
            ed_schedule = sha(E17 / "participation_schedule.json")
            rsr_manifest_hash = sha(CANON42 / "manifest.json")
            ed_manifest_hash = sha(E17 / "manifest.json")
            rsr_payload_hash = sha(CANON42 / "server_visible_update_hashes.json")
            ed_payload_hash = sha(E17 / "server_visible_update_hashes.json")
            protocol_hash = seed42["manifest"]["protocol_spec_sha256"]
            attack_hash = seed42["manifest"]["attack_feature_spec_sha256"]
        else:
            rsr_manifest = json.loads((BASE / f"seed{seed}/rsr/manifest.json").read_text(encoding="utf-8"))
            ed_manifest = json.loads((BASE / f"seed{seed}/ed_om_rsr/manifest.json").read_text(encoding="utf-8"))
            rsr_schedule = rsr_manifest["schedule_sha256"]
            ed_schedule = ed_manifest["schedule_sha256"]
            rsr_manifest_hash = sha(BASE / f"seed{seed}/rsr/manifest.json")
            ed_manifest_hash = sha(BASE / f"seed{seed}/ed_om_rsr/manifest.json")
            rsr_payload_hash = rsr_manifest["payload_and_update_hash"]
            ed_payload_hash = ed_manifest["payload_and_update_hash"]
            protocol_hash = ed_manifest["protocol_spec_sha256"]
            attack_hash = ed_manifest["attack_spec_sha256"]
        rsr_comm = next(row for row in communication_rows if row["Seed"] == seed and row["Method"] == "RSR")
        ed_comm = next(row for row in communication_rows if row["Seed"] == seed and row["Method"] == "ED-OM-RSR")
        comm_pass, ratio = communication_preserved(rsr_comm["TotalSerializedBytes"], ed_comm["TotalSerializedBytes"])
        pair_audit_rows.append({
            "Seed": seed,
            "ScheduleMatched": rsr_schedule == ed_schedule,
            "MaskMatchedReleases": sum(bool(row["MaskMatch"]) for row in release_rows),
            "MaskTotalReleases": len(release_rows),
            "MaskMatched": all(bool(row["MaskMatch"]) for row in release_rows),
            "RetainedCountMatchedReleases": sum(bool(row["RetainedCountMatch"]) for row in release_rows),
            "RetainedCountMatched": all(bool(row["RetainedCountMatch"]) for row in release_rows),
            "SerializedBytesExactReleases": sum(bool(row["SerializedBytesExact"]) for row in release_rows),
            "SerializedBytesExact": all(bool(row["SerializedBytesExact"]) for row in release_rows),
            "RSRBytes": int(rsr_comm["TotalSerializedBytes"]),
            "EDOMBytes": int(ed_comm["TotalSerializedBytes"]),
            "EDOMToRSRRatio": ratio,
            "CommunicationGatePass": comm_pass,
            "ProtocolSpecSHA256": protocol_hash,
            "AttackSpecSHA256": attack_hash,
            "ProtocolSpecMatched": protocol_hash == PROTOCOL_SHA256,
            "AttackSpecMatched": attack_hash == ATTACK_SHA256,
            "PropertyLabelSHA256": label_hashes[seed],
            "OOFFoldSHA256": fold_hashes[seed],
            "RSRManifestSHA256": rsr_manifest_hash,
            "EDOMManifestSHA256": ed_manifest_hash,
            "RSRServerVisiblePayloadHash": rsr_payload_hash,
            "EDOMServerVisiblePayloadHash": ed_payload_hash,
            "ValidPair": (
                rsr_schedule == ed_schedule
                and all(bool(row["MaskMatch"]) for row in release_rows)
                and all(bool(row["RetainedCountMatch"]) for row in release_rows)
                and comm_pass
                and protocol_hash == PROTOCOL_SHA256
                and attack_hash == ATTACK_SHA256
            ),
        })
    if not all(row["ValidPair"] for row in pair_audit_rows):
        raise RuntimeError("invalid paired seed cannot enter aggregate")

    privacy_lookup = {(int(row["Seed"]), row["Method"]): row for row in privacy_rows}
    utility_lookup = {(int(row["Seed"]), row["Method"]): row for row in utility_rows}
    comm_lookup = {(int(row["Seed"]), row["Method"]): row for row in communication_rows}
    reductions = []
    release1_reductions = []
    gaps = []
    ed_better = 0
    release1_better = 0
    extreme_better = 0
    for seed in SEEDS:
        rsr = privacy_lookup[(seed, "RSR")]
        ed = privacy_lookup[(seed, "ED-OM-RSR")]
        reduction = excess_reduction(rsr["PeriodicityFullAUCsep"], ed["PeriodicityFullAUCsep"])
        release_reduction = excess_reduction(rsr["PeriodicityRelease1AUCsep"], ed["PeriodicityRelease1AUCsep"])
        reductions.append(reduction)
        release1_reductions.append(release_reduction)
        gap = ed["PeriodicityFullAUCsep"] - rsr["PeriodicityFullAUCsep"]
        gaps.append(gap)
        ed_better += int(gap < 0.0)
        release1_better += int(ed["PeriodicityRelease1AUCsep"] < rsr["PeriodicityRelease1AUCsep"])
        extreme_better += int(ed["ExtremeFullAUCsep"] < rsr["ExtremeFullAUCsep"])
        for row in (rsr, ed):
            row["PairedPeriodicityExcessReduction"] = "" if reduction is None else reduction
            row["PairedRelease1ExcessReduction"] = "" if release_reduction is None else release_reduction
            row["EDOMBetterFullAdaptive"] = gap < 0.0
            row["EDOMBetterRelease1"] = ed["PeriodicityRelease1AUCsep"] < rsr["PeriodicityRelease1AUCsep"]

    for seed in SEEDS:
        rsr = utility_lookup[(seed, "RSR")]
        ed = utility_lookup[(seed, "ED-OM-RSR")]
        for key in ("MAE", "RMSE", "MAPE"):
            change = (ed[key] - rsr[key]) / rsr[key]
            rsr[f"Paired{key}ChangePct"] = change * 100.0
            ed[f"Paired{key}ChangePct"] = change * 100.0
        warning = (
            ed["PairedMAEChangePct"] > 10.0
            or ed["PairedRMSEChangePct"] > 10.0
        )
        rsr["StabilityWarning"] = warning
        ed["StabilityWarning"] = warning

    for seed in SEEDS:
        rsr = comm_lookup[(seed, "RSR")]
        ed = comm_lookup[(seed, "ED-OM-RSR")]
        passed, ratio = communication_preserved(rsr["TotalSerializedBytes"], ed["TotalSerializedBytes"])
        for row in (rsr, ed):
            row["EDOMToRSRRatio"] = ratio
            row["PairedCommunicationPass"] = passed
            row["CompressionRatioVsDense"] = row["DenseBytes"] / row["TotalSerializedBytes"]
            row["ByteReductionVsDensePct"] = 100.0 * (1.0 - row["TotalSerializedBytes"] / row["DenseBytes"])

    privacy_summary = summarize(
        privacy_rows,
        ("RSR", "ED-OM-RSR"),
        (
            "ExtremeFullAUCsep",
            "PeriodicityStandardAUCsep",
            "PeriodicityRelease1AUCsep",
            "PeriodicityRelease2AUCsep",
            "PeriodicityFullAUCsep",
        ),
    )
    utility_summary = summarize(utility_rows, ("RSR", "ED-OM-RSR"), ("MAE", "RMSE", "MAPE"))
    communication_summary = summarize(
        communication_rows,
        ("RSR", "ED-OM-RSR"),
        ("TotalSerializedBytes", "PayloadBytes", "MetadataBytes", "CompressionRatioVsDense", "ByteReductionVsDensePct"),
    )

    valid_reductions = [value for value in reductions if value is not None]
    valid_release1 = [value for value in release1_reductions if value is not None]
    mean_paired_reduction = float(np.mean(valid_reductions)) if valid_reductions else float("nan")
    mean_release1_reduction = float(np.mean(valid_release1)) if valid_release1 else float("nan")
    rsr_periodic_mean, rsr_periodic_std = sample_mean_std([
        privacy_lookup[(seed, "RSR")]["PeriodicityFullAUCsep"] for seed in SEEDS
    ])
    ed_periodic_mean, ed_periodic_std = sample_mean_std([
        privacy_lookup[(seed, "ED-OM-RSR")]["PeriodicityFullAUCsep"] for seed in SEEDS
    ])
    reduction_from_means = excess_reduction(rsr_periodic_mean, ed_periodic_mean)
    privacy_judgment = replication_judgment(ed_periodic_mean, reductions, ed_better, gaps)

    mae_changes = [utility_lookup[(seed, "ED-OM-RSR")]["PairedMAEChangePct"] / 100.0 for seed in SEEDS]
    rmse_changes = [utility_lookup[(seed, "ED-OM-RSR")]["PairedRMSEChangePct"] / 100.0 for seed in SEEDS]
    mape_changes = [utility_lookup[(seed, "ED-OM-RSR")]["PairedMAPEChangePct"] / 100.0 for seed in SEEDS]
    mean_mae_change = float(np.mean(mae_changes))
    mean_rmse_change = float(np.mean(rmse_changes))
    mean_mape_change = float(np.mean(mape_changes))
    utility_preserved = mean_mae_change <= 0.05 and mean_rmse_change <= 0.05
    stability_warning_seeds = [
        seed for seed in SEEDS if utility_lookup[(seed, "ED-OM-RSR")]["StabilityWarning"]
    ]
    communication_ok = all(row["CommunicationGatePass"] for row in pair_audit_rows)
    final_judgment = privacy_judgment
    if not utility_preserved or not communication_ok:
        final_judgment = (
            "PARTIALLY REPLICATED"
            if privacy_judgment in ("STRONGLY REPLICATED", "REPLICATED")
            else privacy_judgment
        )
    ready = final_judgment in ("STRONGLY REPLICATED", "REPLICATED") and utility_preserved and communication_ok
    proceed = "YES" if ready else "NO"

    if gaps[1] >= 0.0 and gaps[2] >= 0.0 and gaps[0] < 0.0:
        seed42_favorable = "YES"
    elif (
        reductions[0] is not None
        and all(value is not None for value in reductions[1:])
        and reductions[0] > max(reductions[1:]) + 0.20
    ):
        seed42_favorable = "POSSIBLE"
    else:
        seed42_favorable = "NO EVIDENCE"

    periodic_status = (
        "YES" if privacy_judgment in ("STRONGLY REPLICATED", "REPLICATED")
        else "PARTIAL" if privacy_judgment == "PARTIALLY REPLICATED" else "NO"
    )
    release1_status = (
        "YES" if mean_release1_reduction >= 0.50 and release1_better >= 2
        else "PARTIAL" if mean_release1_reduction > 0.0 and release1_better >= 2
        else "NO"
    )
    rsr_extreme_mean, rsr_extreme_std = sample_mean_std([
        privacy_lookup[(seed, "RSR")]["ExtremeFullAUCsep"] for seed in SEEDS
    ])
    ed_extreme_mean, ed_extreme_std = sample_mean_std([
        privacy_lookup[(seed, "ED-OM-RSR")]["ExtremeFullAUCsep"] for seed in SEEDS
    ])
    extreme_status = (
        "YES" if ed_extreme_mean < rsr_extreme_mean and extreme_better >= 2
        else "PARTIAL" if ed_extreme_mean < rsr_extreme_mean or extreme_better >= 1
        else "NO"
    )

    dedicated = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_exact_dimension_orthogonal_mixing.py",
            "tests/test_ed_om_rsr_attack.py",
            "tests/test_ed_om_rsr_multiseed.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    dedicated_text = dedicated.stdout + dedicated.stderr
    dedicated_passed, dedicated_failed = e11.parse_tests(dedicated_text)
    full = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    full_text = full.stdout + full.stderr
    full_passed, full_failed = e11.parse_tests(full_text)
    if dedicated.returncode or full.returncode or dedicated_failed or full_failed:
        raise RuntimeError(f"Experiment 18 tests failed: dedicated={dedicated_failed}, full={full_failed}")

    def fmt(value: float) -> str:
        return f"{float(value):.6f}"

    def pm(mean: float, std: float) -> str:
        return f"{mean:.6f} ± {std:.6f}"

    rsr_r1_mean, rsr_r1_std = sample_mean_std([
        privacy_lookup[(seed, "RSR")]["PeriodicityRelease1AUCsep"] for seed in SEEDS
    ])
    ed_r1_mean, ed_r1_std = sample_mean_std([
        privacy_lookup[(seed, "ED-OM-RSR")]["PeriodicityRelease1AUCsep"] for seed in SEEDS
    ])
    rsr_r2_mean, rsr_r2_std = sample_mean_std([
        privacy_lookup[(seed, "RSR")]["PeriodicityRelease2AUCsep"] for seed in SEEDS
    ])
    ed_r2_mean, ed_r2_std = sample_mean_std([
        privacy_lookup[(seed, "ED-OM-RSR")]["PeriodicityRelease2AUCsep"] for seed in SEEDS
    ])
    utility_means = {}
    for method_name in ("RSR", "ED-OM-RSR"):
        utility_means[method_name] = {
            key: sample_mean_std([utility_lookup[(seed, method_name)][key] for seed in SEEDS])
            for key in ("MAE", "RMSE", "MAPE")
        }
    comm_means = {
        method_name: sample_mean_std([
            comm_lookup[(seed, method_name)]["TotalSerializedBytes"] for seed in SEEDS
        ])
        for method_name in ("RSR", "ED-OM-RSR")
    }
    mae_wins = sum(
        utility_lookup[(seed, "ED-OM-RSR")]["MAE"] < utility_lookup[(seed, "RSR")]["MAE"]
        for seed in SEEDS
    )
    rmse_wins = sum(
        utility_lookup[(seed, "ED-OM-RSR")]["RMSE"] < utility_lookup[(seed, "RSR")]["RMSE"]
        for seed in SEEDS
    )

    terminal_summary = f"""============================================================
EXPERIMENT 18 — ED-OM-RSR MULTI-SEED REPLICATION
============================================================

Dataset:
PEMS08

Seeds:
42 / 123 / 2026

p:
0.5

Protocol changed:
NO

Attack spec changed:
NO

DP:
NO

------------------------------------------------------------
PAIRING AUDIT
------------------------------------------------------------

Seed42:
Schedule matched:
PASS
Mask matched:
PASS
Retained count matched:
PASS

Seed123:
Schedule matched:
PASS
Mask matched:
PASS
Retained count matched:
PASS

Seed2026:
Schedule matched:
PASS
Mask matched:
PASS
Retained count matched:
PASS

------------------------------------------------------------
PERIODICITY FULL ADAPTIVE
------------------------------------------------------------

Seed42:
RSR = {fmt(privacy_lookup[(42, "RSR")]["PeriodicityFullAUCsep"])}
ED-OM = {fmt(privacy_lookup[(42, "ED-OM-RSR")]["PeriodicityFullAUCsep"])}
Excess reduction = {100.0 * reductions[0]:.2f} %

Seed123:
RSR = {fmt(privacy_lookup[(123, "RSR")]["PeriodicityFullAUCsep"])}
ED-OM = {fmt(privacy_lookup[(123, "ED-OM-RSR")]["PeriodicityFullAUCsep"])}
Excess reduction = {"NA" if reductions[1] is None else f"{100.0 * reductions[1]:.2f} %"}

Seed2026:
RSR = {fmt(privacy_lookup[(2026, "RSR")]["PeriodicityFullAUCsep"])}
ED-OM = {fmt(privacy_lookup[(2026, "ED-OM-RSR")]["PeriodicityFullAUCsep"])}
Excess reduction = {"NA" if reductions[2] is None else f"{100.0 * reductions[2]:.2f} %"}

3-seed:

RSR =
{pm(rsr_periodic_mean, rsr_periodic_std)}

ED-OM =
{pm(ed_periodic_mean, ed_periodic_std)}

Mean paired excess reduction:
{100.0 * mean_paired_reduction:.2f} %

ED-OM better:
{ed_better} / 3 seeds

------------------------------------------------------------
RELEASE-1 PERIODICITY
------------------------------------------------------------

Seed42:
RSR = {fmt(privacy_lookup[(42, "RSR")]["PeriodicityRelease1AUCsep"])}
ED-OM = {fmt(privacy_lookup[(42, "ED-OM-RSR")]["PeriodicityRelease1AUCsep"])}

Seed123:
RSR = {fmt(privacy_lookup[(123, "RSR")]["PeriodicityRelease1AUCsep"])}
ED-OM = {fmt(privacy_lookup[(123, "ED-OM-RSR")]["PeriodicityRelease1AUCsep"])}

Seed2026:
RSR = {fmt(privacy_lookup[(2026, "RSR")]["PeriodicityRelease1AUCsep"])}
ED-OM = {fmt(privacy_lookup[(2026, "ED-OM-RSR")]["PeriodicityRelease1AUCsep"])}

3-seed summary:
RSR = {pm(rsr_r1_mean, rsr_r1_std)}
ED-OM = {pm(ed_r1_mean, ed_r1_std)}
Mean paired excess reduction = {100.0 * mean_release1_reduction:.2f} %

------------------------------------------------------------
EXTREME EVENT
------------------------------------------------------------

RSR Full Adaptive:
{pm(rsr_extreme_mean, rsr_extreme_std)}

ED-OM Full Transform-Aware:
{pm(ed_extreme_mean, ed_extreme_std)}

------------------------------------------------------------
UTILITY
------------------------------------------------------------

RSR:

MAE = {pm(*utility_means["RSR"]["MAE"])}
RMSE = {pm(*utility_means["RSR"]["RMSE"])}
MAPE = {pm(*utility_means["RSR"]["MAPE"])}

ED-OM:

MAE = {pm(*utility_means["ED-OM-RSR"]["MAE"])}
RMSE = {pm(*utility_means["ED-OM-RSR"]["RMSE"])}
MAPE = {pm(*utility_means["ED-OM-RSR"]["MAPE"])}

Mean MAE degradation:
{100.0 * mean_mae_change:.2f} %

Mean RMSE degradation:
{100.0 * mean_rmse_change:.2f} %

ED-OM MAE wins:
{mae_wins} / 3

ED-OM RMSE wins:
{rmse_wins} / 3

Stability warning seeds:
{stability_warning_seeds if stability_warning_seeds else "NONE"}

------------------------------------------------------------
COMMUNICATION
------------------------------------------------------------

Seed42:
RSR bytes = {comm_lookup[(42, "RSR")]["TotalSerializedBytes"]}
ED-OM bytes = {comm_lookup[(42, "ED-OM-RSR")]["TotalSerializedBytes"]}

Seed123:
RSR bytes = {comm_lookup[(123, "RSR")]["TotalSerializedBytes"]}
ED-OM bytes = {comm_lookup[(123, "ED-OM-RSR")]["TotalSerializedBytes"]}

Seed2026:
RSR bytes = {comm_lookup[(2026, "RSR")]["TotalSerializedBytes"]}
ED-OM bytes = {comm_lookup[(2026, "ED-OM-RSR")]["TotalSerializedBytes"]}

Communication preserved:
{"YES" if communication_ok else "NO"}

------------------------------------------------------------
TESTS
------------------------------------------------------------

Dedicated:
{dedicated_passed} passed
{dedicated_failed} failed

Full:
{full_passed} passed
{full_failed} failed

------------------------------------------------------------
FINAL JUDGMENT
------------------------------------------------------------

Periodicity privacy replication:
{periodic_status}

Release-1 leakage suppression replicated:
{release1_status}

Extreme mitigation replicated:
{extreme_status}

Utility preserved:
{"YES" if utility_preserved else "NO"}

Communication preserved:
{"YES" if communication_ok else "NO"}

Seed42 appears unusually favorable:
{seed42_favorable}

ED-OM-RSR MULTI-SEED:
{final_judgment}

ED-OM-RSR READY AS MAIN PAPER METHOD:
{"YES" if ready else "NO"}

PROCEED TO METR-LA VALIDATION:
{proceed}

============================================================
"""

    table1 = [
        {
            "Method": method_name,
            "ExtremeFullAdaptiveMean": rsr_extreme_mean if method_name == "RSR" else ed_extreme_mean,
            "ExtremeFullAdaptiveSampleStd": rsr_extreme_std if method_name == "RSR" else ed_extreme_std,
            "PeriodicityFullAdaptiveMean": rsr_periodic_mean if method_name == "RSR" else ed_periodic_mean,
            "PeriodicityFullAdaptiveSampleStd": rsr_periodic_std if method_name == "RSR" else ed_periodic_std,
            "MAEMean": utility_means[method_name]["MAE"][0],
            "MAESampleStd": utility_means[method_name]["MAE"][1],
            "RMSEMean": utility_means[method_name]["RMSE"][0],
            "RMSESampleStd": utility_means[method_name]["RMSE"][1],
            "BytesMean": comm_means[method_name][0],
            "BytesSampleStd": comm_means[method_name][1],
        }
        for method_name in ("RSR", "ED-OM-RSR")
    ]
    table2 = [
        {
            "Method": method_name,
            "Release1PeriodicityMean": rsr_r1_mean if method_name == "RSR" else ed_r1_mean,
            "Release1PeriodicitySampleStd": rsr_r1_std if method_name == "RSR" else ed_r1_std,
            "Release2PeriodicityMean": rsr_r2_mean if method_name == "RSR" else ed_r2_mean,
            "Release2PeriodicitySampleStd": rsr_r2_std if method_name == "RSR" else ed_r2_std,
            "FullAdaptiveMean": rsr_periodic_mean if method_name == "RSR" else ed_periodic_mean,
            "FullAdaptiveSampleStd": rsr_periodic_std if method_name == "RSR" else ed_periodic_std,
        }
        for method_name in ("RSR", "ED-OM-RSR")
    ]

    report_rows = []
    for seed in SEEDS:
        rsr_p, ed_p = privacy_lookup[(seed, "RSR")], privacy_lookup[(seed, "ED-OM-RSR")]
        rsr_u, ed_u = utility_lookup[(seed, "RSR")], utility_lookup[(seed, "ED-OM-RSR")]
        rsr_c, ed_c = comm_lookup[(seed, "RSR")], comm_lookup[(seed, "ED-OM-RSR")]
        report_rows.append(
            f"| {seed} | {rsr_p['PeriodicityFullAUCsep']:.3f} | {ed_p['PeriodicityFullAUCsep']:.3f} | "
            f"{'NA' if reductions[SEEDS.index(seed)] is None else f'{100*reductions[SEEDS.index(seed)]:.2f}%'} | "
            f"{rsr_p['PeriodicityRelease1AUCsep']:.3f} | {ed_p['PeriodicityRelease1AUCsep']:.3f} | "
            f"{rsr_p['ExtremeFullAUCsep']:.3f} | {ed_p['ExtremeFullAUCsep']:.3f} | "
            f"{rsr_u['MAE']:.3f} | {ed_u['MAE']:.3f} | {rsr_c['TotalSerializedBytes']} | {ed_c['TotalSerializedBytes']} |"
        )
    report = f"""# Experiment 18 — Multi-Seed Replication of ED-OM-RSR

This experiment is a strict replication, not a mechanism or attacker-development study. The frozen ED-OM-RSR protocol and attack specification were reused unchanged. ED-OM-RSR does not claim epsilon-delta differential privacy; it measures empirical resistance to property inference under a public-protocol honest-but-curious server.

The formal Experiment 18 run manifest was frozen before the four new victim trainings and all new AUCs at {RUN_MANIFEST_SHA256}. Seed42 reused the Experiment 17 and canonical RSR artifacts; only the paired RSR/ED-OM-RSR runs for seed123 and seed2026 were trained.

## Table E18-1

| Method | Extreme Full Adaptive ↓ | Periodicity Full Adaptive ↓ | MAE ↓ | RMSE ↓ | Bytes ↓ |
|---|---:|---:|---:|---:|---:|
| RSR | {rsr_extreme_mean:.3f} ± {rsr_extreme_std:.3f} | {rsr_periodic_mean:.3f} ± {rsr_periodic_std:.3f} | {utility_means["RSR"]["MAE"][0]:.3f} ± {utility_means["RSR"]["MAE"][1]:.3f} | {utility_means["RSR"]["RMSE"][0]:.3f} ± {utility_means["RSR"]["RMSE"][1]:.3f} | {comm_means["RSR"][0]:.0f} ± {comm_means["RSR"][1]:.0f} |
| ED-OM-RSR | {ed_extreme_mean:.3f} ± {ed_extreme_std:.3f} | {ed_periodic_mean:.3f} ± {ed_periodic_std:.3f} | {utility_means["ED-OM-RSR"]["MAE"][0]:.3f} ± {utility_means["ED-OM-RSR"]["MAE"][1]:.3f} | {utility_means["ED-OM-RSR"]["RMSE"][0]:.3f} ± {utility_means["ED-OM-RSR"]["RMSE"][1]:.3f} | {comm_means["ED-OM-RSR"][0]:.0f} ± {comm_means["ED-OM-RSR"][1]:.0f} |

## Table E18-2

| Method | Release-1 Periodicity ↓ | Release-2 Periodicity ↓ | Full Adaptive ↓ |
|---|---:|---:|---:|
| RSR | {rsr_r1_mean:.3f} ± {rsr_r1_std:.3f} | {rsr_r2_mean:.3f} ± {rsr_r2_std:.3f} | {rsr_periodic_mean:.3f} ± {rsr_periodic_std:.3f} |
| ED-OM-RSR | {ed_r1_mean:.3f} ± {ed_r1_std:.3f} | {ed_r2_mean:.3f} ± {ed_r2_std:.3f} | {ed_periodic_mean:.3f} ± {ed_periodic_std:.3f} |

All standard deviations are seed-level sample standard deviations (ddof=1). The analysis treats these as three stochastic replications over a 50-client protocol, not as 150 independent clients.

## Per-seed results

| Seed | RSR Periodicity Full | ED Full | Excess reduction | RSR R1 | ED R1 | RSR Extreme | ED Extreme | RSR MAE | ED MAE | RSR bytes | ED bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(report_rows) + f"""

The formal mean of per-seed paired Periodicity excess reductions is {100.0 * mean_paired_reduction:.2f}%. The secondary reduction calculated from the two mean AUC-separability values is {"NA" if reduction_from_means is None else f"{100.0 * reduction_from_means:.2f}%"}. These are related but not identical statistics. ED-OM is lower than RSR in {ed_better}/3 seeds.

## Pairing, utility, and communication

Every seed passed schedule equality, 100/100 bitwise mask equality, 100/100 retained-count equality, frozen protocol/attack hashes, and the per-seed communication gate. Actual RSR and ED-OM serialized byte totals are exactly equal within all three seeds.

Mean paired MAE/RMSE changes are {100.0 * mean_mae_change:.2f}% and {100.0 * mean_rmse_change:.2f}%; utility preserved: **{"YES" if utility_preserved else "NO"}**. ED-OM wins MAE/RMSE in {mae_wins}/3 and {rmse_wins}/3 seeds. Stability warning seeds (>10% MAE or RMSE degradation): **{stability_warning_seeds if stability_warning_seeds else "NONE"}**.

## Answers to the eight required questions

1. **Q1 — Does Periodicity Full Transform-Aware improvement replicate? {periodic_status}.** Final privacy judgment: {privacy_judgment}.
2. **Q2 — Is Release-1 retained-value leakage stably reduced? {release1_status}.** ED-OM is lower in {release1_better}/3 seeds; mean paired excess reduction is {100.0 * mean_release1_reduction:.2f}%.
3. **Q3 — Is Extreme Event mitigation retained? {extreme_status}.** RSR/ED means are {rsr_extreme_mean:.3f}/{ed_extreme_mean:.3f}; ED is lower in {extreme_better}/3 seeds.
4. **Q4 — Is three-seed utility degradation within 5%? {"YES" if utility_preserved else "NO"}.**
5. **Q5 — Does ED-OM preserve canonical RSR communication? {"YES" if communication_ok else "NO"}.**
6. **Q6 — Was seed42 unusually favorable? {seed42_favorable}.**
7. **Q7 — Is ED-OM-RSR ready as the main paper method? {"YES" if ready else "NO"}.**
8. **Q8 — Proceed to METR-LA cross-dataset validation? {proceed}.** Experiment 19 was not started.

Dedicated tests: {dedicated_passed} passed, 0 failed. Full suite: {full_passed} passed, 0 failed.

**ED-OM-RSR MULTI-SEED: {final_judgment}**

**PROCEED TO METR-LA VALIDATION: {proceed}**
"""

    stage = BASE / f".summary_staging_{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    attack_root = stage / "attack"
    attack_root.mkdir()
    for seed in (123, 2026):
        seed_dir = attack_root / f"seed{seed}"
        seed_dir.mkdir()
        attack = all_attack_artifacts[seed]
        write_rows(seed_dir / "attack_metrics.csv", attack["metrics"])
        write_rows(seed_dir / "attack_bootstrap_ci.csv", attack["bootstrap"])
        write_rows(seed_dir / "attack_oof_predictions.csv", attack["predictions"])
        write_rows(seed_dir / "attack_features.csv", attack["features"])
        provenance_fields = list(dict.fromkeys(
            key for row in attack["provenance"] for key in row
        ))
        normalized_provenance = [
            {key: row.get(key, "") for key in provenance_fields}
            for row in attack["provenance"]
        ]
        write_rows(seed_dir / "attack_feature_provenance.csv", normalized_provenance)
        write_rows(seed_dir / "property_labels_and_folds.csv", label_records_by_seed[seed])
        write_json(seed_dir / "attack_schema.json", {
            "seed": seed,
            "feature_spec_sha256": ATTACK_SHA256,
            "property_spec_sha256": PROPERTY_SPEC_SHA256,
            "conditions": {name: list(names) for name, names in attack["schemas"].items()},
            "property_label_sha256": label_hashes[seed],
            "oof_fold_sha256": fold_hashes[seed],
        })
    write_rows(stage / "per_seed_utility.csv", utility_rows)
    write_rows(stage / "per_seed_privacy.csv", privacy_rows)
    write_rows(stage / "per_seed_communication.csv", communication_rows)
    write_rows(stage / "per_seed_pairing_audit.csv", pair_audit_rows)
    write_rows(stage / "per_release_pairing_audit.csv", all_pair_rows)
    write_rows(stage / "privacy_summary.csv", privacy_summary)
    write_rows(stage / "utility_summary.csv", utility_summary)
    write_rows(stage / "communication_summary.csv", communication_summary)
    write_rows(stage / "table_e18_1.csv", table1)
    write_rows(stage / "table_e18_2.csv", table2)
    (stage / "FINAL_TERMINAL_SUMMARY.txt").write_text(terminal_summary, encoding="utf-8")
    (stage / "test_results.txt").write_text(
        "=== PRETRAINING DEDICATED ===\n"
        + PRETEST.read_text(encoding="utf-8")
        + "\n=== POST-EVALUATION DEDICATED ===\n"
        + dedicated_text
        + "\n=== FULL SUITE ===\n"
        + full_text,
        encoding="utf-8",
    )

    pointer = {
        "seed": 42,
        "policy": "reference only; no copy and no retraining",
        "canonical_rsr": str(CANON42),
        "implicit_mask_communication": str(IMR42),
        "ed_om_rsr": str(E17),
        "canonical_rsr_manifest_sha256": sha(CANON42 / "manifest.json"),
        "ed_om_rsr_manifest_sha256": sha(E17 / "manifest.json"),
        "property_label_sha256": label_hashes[42],
        "oof_fold_sha256": fold_hashes[42],
    }
    pointer_text = json.dumps(pointer, indent=2, sort_keys=True) + "\n"
    output_hashes = {
        str(path.relative_to(stage)): sha(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "multiseed_manifest.json"
    }
    final_manifest = {
        "experiment": 18,
        "name": "Multi-Seed Replication of ED-OM-RSR",
        "dataset": "PEMS08",
        "seeds": list(SEEDS),
        "p": 0.5,
        "new_trainings": [
            {"seed": 123, "method": "RSR"},
            {"seed": 123, "method": "ED-OM-RSR"},
            {"seed": 2026, "method": "RSR"},
            {"seed": 2026, "method": "ED-OM-RSR"},
        ],
        "new_training_count": 4,
        "seed42_retrained": False,
        "protocol_changed": False,
        "attack_spec_changed": False,
        "DP": False,
        "run_manifest_sha256": RUN_MANIFEST_SHA256,
        "protocol_spec_sha256": PROTOCOL_SHA256,
        "attack_spec_sha256": ATTACK_SHA256,
        "property_spec_sha256": PROPERTY_SPEC_SHA256,
        "pairing_audit": pair_audit_rows,
        "property_label_hashes": label_hashes,
        "oof_fold_hashes": fold_hashes,
        "aggregation": {
            "unit": "seed",
            "seeds": 3,
            "pooled_client_claim": False,
            "sample_std_ddof": 1,
            "mean_paired_periodicity_excess_reduction": mean_paired_reduction,
            "periodicity_excess_reduction_from_mean_aucs": reduction_from_means,
            "mean_paired_release1_excess_reduction": mean_release1_reduction,
        },
        "privacy": {
            "rsr_periodicity_full_mean": rsr_periodic_mean,
            "rsr_periodicity_full_sample_std": rsr_periodic_std,
            "ed_periodicity_full_mean": ed_periodic_mean,
            "ed_periodicity_full_sample_std": ed_periodic_std,
            "ed_better_seeds": ed_better,
            "release1_better_seeds": release1_better,
            "extreme_better_seeds": extreme_better,
            "privacy_judgment": privacy_judgment,
            "periodicity_status": periodic_status,
            "release1_status": release1_status,
            "extreme_status": extreme_status,
            "seed42_unusually_favorable": seed42_favorable,
        },
        "utility": {
            "mean_mae_change_vs_paired_rsr": mean_mae_change,
            "mean_rmse_change_vs_paired_rsr": mean_rmse_change,
            "mean_mape_change_vs_paired_rsr": mean_mape_change,
            "mae_wins": mae_wins,
            "rmse_wins": rmse_wins,
            "stability_warning_seeds": stability_warning_seeds,
            "preserved": utility_preserved,
        },
        "communication": {
            "preserved_all_seeds": communication_ok,
            "exact_byte_equality_all_seeds": all(row["SerializedBytesExact"] for row in pair_audit_rows),
        },
        "judgment": final_judgment,
        "ready_as_main_paper_method": ready,
        "proceed_to_metr_la_validation": proceed,
        "automatic_metr_la_started": False,
        "tests": {
            "pretraining": pretests,
            "dedicated": {"passed": dedicated_passed, "failed": dedicated_failed},
            "full": {"passed": full_passed, "failed": full_failed},
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "source_hashes": {
            "experiments/ed_om_rsr_multiseed_common.py": sha(ROOT / "experiments/ed_om_rsr_multiseed_common.py"),
            "experiments/exp_ed_om_rsr_multiseed_train.py": sha(ROOT / "experiments/exp_ed_om_rsr_multiseed_train.py"),
            "experiments/finalize_ed_om_rsr_multiseed.py": sha(ROOT / "experiments/finalize_ed_om_rsr_multiseed.py"),
            "tests/test_ed_om_rsr_multiseed.py": sha(ROOT / "tests/test_ed_om_rsr_multiseed.py"),
        },
        "formal_input_hashes": run_manifest["formal_inputs"],
        "seed42_reference_pointer_sha256": text_sha(pointer_text),
        "report_sha256": text_sha(report),
        "output_hashes": output_hashes,
        "historical_artifacts_modified": False,
    }
    write_json(stage / "multiseed_manifest.json", final_manifest)
    stage.rename(SUMMARY)
    seed42_dir = BASE / "seed42"
    seed42_dir.mkdir(exist_ok=False)
    (seed42_dir / "reference_manifest.json").write_text(pointer_text, encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    if sha(REPORT) != final_manifest["report_sha256"]:
        raise RuntimeError("Experiment 18 report hash mismatch")
    print(terminal_summary, end="")


if __name__ == "__main__":
    run()

