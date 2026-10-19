from __future__ import annotations

"""Experiment 9 continuation: locked-seed, trajectory-consistent utility and attack evidence."""

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from attacks.spectral_leakage.collector import UpdateCollector, UpdateRecord
from experiments.common import configure_logging, save_config, write_rows
from experiments.exp_ellipsoidal_clipping_sanity import (
    _build_clients_and_eval,
    _environment_snapshot,
    _evaluate_partitions,
    _prepare_data,
)
from experiments.exp_matched_reduced_release_frontier import (
    DPFEDAVG_CLIP,
    calibrate_dpfedavg,
    calibrate_pjsdp,
    load_and_verify_schedule,
)
from experiments.exp_privacy_utility_frontier import DiagnosticDPFedAvg, _flat_cosine, _flat_norm
from experiments.exp_public_ellipsoid_dp_sanity import (
    PublicGeometryReleaseMechanism,
    _cosine,
    _global_norm,
)
from experiments.exp_representative_privacy_attack import (
    ALLOWED_LAYERS,
    ATTACKS,
    ATTACK_CV_SEED,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CLIENTS,
    CLIENTS_PER_ROUND,
    DELTA,
    EXPECTED_FEATURE_SPEC_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    LOCAL_EPOCHS,
    PROPERTIES,
    PROPERTY_DISPLAY,
    RELEASES_PER_CLIENT,
    ROUNDS,
    _plot_style,
    _property_fold_rows,
    build_property_labels,
    decide,
    evaluate_attack,
    extract_feature_families,
    make_folds,
    validate_feature_spec,
)
from experiments.exp_spectral_norm_audit import _load_reference_config, _resolve_device, _set_seed
from federated.stg_server import STGServer
from privacy.clipping import clip_group
from privacy.dp_fedavg import DPUpdate
from privacy.fixed_spectral_projection import FixedSpectralProjection
from spectral.spectral_groups import GROUPS, inverse_groups, transform_update


BASE_DP_NOISE_SEED = 20260903
EXPECTED_POLICY_SHA256 = "cd684b0905a405eb6a02b737d7e145cb01ec6bd8da8161d4899d9d35feee96d5"
VICTIMS = (
    ("fedavg_r2", "FedAvg", None),
    ("dp_fedavg_eps2048", "DP-FedAvg", 2048),
    ("pjsdp_eps2048", "PJSDP", 2048),
    ("dp_fedavg_eps1024", "DP-FedAvg", 1024),
    ("pjsdp_eps1024", "PJSDP", 1024),
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--reference_config", type=Path, required=True)
    p.add_argument("--schedule_path", type=Path, required=True)
    p.add_argument("--feature_spec", type=Path, required=True)
    p.add_argument("--noise_policy", type=Path, required=True)
    p.add_argument("--historical_frontier", type=Path, required=True)
    p.add_argument("--legacy_replay", type=Path, required=True)
    p.add_argument("--results_root", type=Path, required=True)
    p.add_argument("--report_path", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    return p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def derive_seed(*, method: str, epsilon: int | None, round_idx: int, client_id: int, group_id: str, tensor_name: str) -> int:
    payload = {
        "base_seed": BASE_DP_NOISE_SEED,
        "method": method,
        "epsilon": "INF" if epsilon is None else int(epsilon),
        "round_idx": int(round_idx),
        "client_id": int(client_id),
        "group_id": str(group_id),
        "tensor_name": str(tensor_name),
    }
    encoded = b"FedSpecShieldLockedDPSeedV1\0" + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
    return seed or 1


def _generator(value: torch.Tensor, **seed_fields: Any) -> torch.Generator:
    return torch.Generator(device=value.device).manual_seed(derive_seed(**seed_fields))


class LockedDiagnosticDPFedAvg(DiagnosticDPFedAvg):
    def __init__(self, clip_norm: float, noise_std: float, epsilon: int) -> None:
        super().__init__(clip_norm, noise_std, torch.Generator().manual_seed(1))
        self.epsilon = int(epsilon)

    def privatize(self, update: Mapping[str, torch.Tensor], round_index: int = -1, client_id: int = -1) -> DPUpdate:
        if round_index < 0 or client_id < 0:
            raise ValueError("locked DP release requires round and client context")
        clipped, pre_norm = clip_group(update, self.clip_norm)
        noisy: dict[str, torch.Tensor] = {}
        noise_update: dict[str, torch.Tensor] = {}
        for name, value in clipped.items():
            generator = _generator(
                value, method="dp_fedavg", epsilon=self.epsilon, round_idx=round_index,
                client_id=client_id, group_id="full_update", tensor_name=name,
            )
            noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator) * self.noise_std
            noise_update[name] = noise
            noisy[name] = value + noise
        post_norm = _flat_norm(clipped)
        factor = min(1.0, self.clip_norm / (float(pre_norm) + 1e-12))
        noise_norm = _flat_norm(noise_update)
        self.diagnostics.append({
            "round": round_index + 1, "client_id": client_id, "pre_clip_norm": float(pre_norm),
            "post_clip_norm": post_norm, "clipping_factor": factor,
            "clipping_distortion": 1.0 - factor if pre_norm > 0 else 0.0,
            "noise_std": self.noise_std, "noise_norm": noise_norm, "signal_norm": post_norm,
            "noise_to_signal_ratio": noise_norm / (post_norm + 1e-12),
            "dp_alignment": _flat_cosine(clipped, noisy),
            "diagnostic_scope": "PRIVATE_INTERNAL_NOT_PART_OF_DP_TRANSCRIPT",
        })
        return DPUpdate(noisy, True, {})


class LockedPublicGeometryReleaseMechanism(PublicGeometryReleaseMechanism):
    def __init__(self, projection: FixedSpectralProjection, bounds: Mapping[str, float], noise_stds: Mapping[str, float], epsilon: int) -> None:
        super().__init__("current", projection, bounds, None, noise_stds, torch.Generator().manual_seed(1))
        self.epsilon = int(epsilon)

    def privatize(self, update: Mapping[str, torch.Tensor], round_index: int = -1, client_id: int = -1) -> DPUpdate:
        if round_index < 0 or client_id < 0:
            raise ValueError("locked PJSDP release requires round and client context")
        projected = self.projection.project_groups(transform_update(update))
        clipped, diagnostics, group_rows = self._clip(projected)
        noisy: dict[str, dict[str, torch.Tensor]] = {}
        for group in GROUPS:
            noisy[group] = {}
            for name, value in clipped[group].items():
                generator = _generator(
                    value, method="pjsdp", epsilon=self.epsilon, round_idx=round_index,
                    client_id=client_id, group_id=group, tensor_name=name,
                )
                noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
                noisy[group][name] = value + noise * self.noise_stds[group]
        released_groups = self.projection.project_groups(noisy)
        noise_groups = {
            group: {name: released_groups[group][name] - clipped[group][name] for name in clipped[group]}
            for group in GROUPS
        }
        signal_norm = _global_norm(clipped)
        noise_norm = _global_norm(noise_groups)
        dp_alignment = _cosine(clipped, released_groups)
        common = {"round": round_index + 1, "client_id": client_id, "method": self.method}
        self.clipping_diagnostics.append({**common, **diagnostics, "dp_alignment": dp_alignment})
        for row in group_rows:
            self.group_diagnostics.append({**common, **row})
        self.noise_signal_diagnostics.append({
            **common, "signal_norm": signal_norm, "noise_norm": noise_norm,
            "noise_to_signal_ratio": noise_norm / (signal_norm + 1e-12),
            "clip_alignment": diagnostics["clip_alignment"], "dp_alignment": dp_alignment,
            "diagnostic_scope": "PRIVATE_INTERNAL_NOT_PART_OF_DP_TRANSCRIPT",
        })
        return DPUpdate(inverse_groups(released_groups, update), True, {})


def _set_deterministic() -> None:
    _set_seed(42)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _tensor_hash(update: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(update):
        value = update[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def released_cache_manifest(records: Sequence[UpdateRecord]) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for release_index, record in enumerate(sorted(records, key=lambda row: (row.round, row.client_id))):
        rows.append({
            "ReleaseIndex": release_index, "Round": record.round, "ClientID": record.client_id,
            "TensorHash": _tensor_hash(record.layers), "UpdateNorm": record.update_norm,
        })
    digest = canonical_hash({"observations": [{k: row[k] for k in ("ReleaseIndex", "Round", "ClientID", "TensorHash")} for row in rows]})
    return rows, digest


def _mechanism(method: str, epsilon: int | None, model: torch.nn.Module, config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    if method == "FedAvg":
        return None, {"AchievedEpsilon": "INF / Non-Private", "Noise": "NONE"}
    assert epsilon is not None
    if method == "DP-FedAvg":
        calibration = calibrate_dpfedavg(float(epsilon))
        return LockedDiagnosticDPFedAvg(DPFEDAVG_CLIP, calibration["noise_std"], epsilon), calibration
    calibration = calibrate_pjsdp(config, float(epsilon))
    projection = FixedSpectralProjection(model.state_dict(), 0.90, 42)
    if projection.mask_hash != str(config["projection_mask_hash"]):
        raise RuntimeError("projection mask hash mismatch")
    bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
    return LockedPublicGeometryReleaseMechanism(projection, bounds, calibration["group_noise_stds"], epsilon), calibration


def victim_run_id(method: str, epsilon: int | None) -> str:
    stem = method.lower().replace("-", "_")
    return f"locked_seed42_{stem}_{'nonprivate' if epsilon is None else f'eps{epsilon}'}_r2"


def run_victim(*, key: str, method: str, epsilon: int | None, output: Path, reference: Mapping[str, Any], prepared: tuple, schedule: Sequence[Sequence[int]], device: torch.device, policy_hash: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    updates_dir = output / "server_visible_updates"
    configure_logging(output / "train.log")
    _set_deterministic()
    model, clients, validation_sets, test_sets = _build_clients_and_eval(reference, prepared, device)
    mechanism, calibration = _mechanism(method, epsilon, model, reference)
    base_run_id = victim_run_id(method, epsilon)
    run_id = f"{base_run_id}_{key}" if key.startswith("repro_") else base_run_id
    config = {
        "VictimRunID": run_id, "experiment": "Locked-Seed Self-Consistent Evidence Set",
        "method": method, "epsilon": "INF / Non-Private" if epsilon is None else epsilon,
        "delta": None if epsilon is None else DELTA, "adjacency": None if epsilon is None else "replacement",
        "seed": 42, "model_seed": 42, "base_dp_noise_seed": None if epsilon is None else BASE_DP_NOISE_SEED,
        "noise_seed_policy_hash": policy_hash, "rounds": ROUNDS, "num_clients": CLIENTS,
        "clients_per_round": CLIENTS_PER_ROUND, "releases_per_client": RELEASES_PER_CLIENT,
        "local_epochs": LOCAL_EPOCHS, "schedule_hash": EXPECTED_SCHEDULE_SHA256,
        "history": 12, "horizon": 12, "batch_size": 64, "learning_rate": 0.01,
        "projection_ratio": None if method != "PJSDP" else 0.90,
        "projection_mask_hash": None if method != "PJSDP" else str(reference["projection_mask_hash"]),
        "clipping": "none" if method == "FedAvg" else ("full_update_l2" if method == "DP-FedAvg" else "current_independent_group_wise"),
        "accountant": "N/A" if method == "FedAvg" else "verified_RDP_no_subsampling_amplification",
        "attack_observation": "server-visible released update used for aggregation",
    }
    core_config_hash = canonical_hash({k: v for k, v in config.items() if k != "VictimRunID"})
    save_config(output / "config.yaml", config)
    collector = UpdateCollector(save_full_updates=True, output_dir=updates_dir)
    server = STGServer(
        model, clients, device, 42, collector=collector, privacy=mechanism,
        topk_ratio=1.0, error_feedback=False, event_tau=0.0, skip_policy="zero",
        report_client_train_loss=True, participation_schedule=schedule,
    )
    losses = []
    started = time.perf_counter()
    for round_index in range(ROUNDS):
        row = server.run_round(round_index, CLIENTS_PER_ROUND, LOCAL_EPOCHS, 0.0)
        loss = float(row["client_train_loss"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss for {run_id}")
        losses.append(loss)
    if len(collector.records) != ROUNDS * CLIENTS_PER_ROUND:
        raise RuntimeError("expected exactly 100 server-visible releases")
    by_client = collector.by_client()
    if set(by_client) != set(range(CLIENTS)) or any(len(rows) != 2 for rows in by_client.values()):
        raise RuntimeError("fixed schedule did not produce two releases per client")
    *_, mean, std = prepared
    validation = _evaluate_partitions(server.model, validation_sets, device, reference, mean, std)
    test = _evaluate_partitions(server.model, test_sets, device, reference, mean, std)
    cache_rows, cache_hash = released_cache_manifest(collector.records)
    for row in cache_rows:
        row.update({"VictimRunID": run_id, "Method": method, "Epsilon": "INF / Non-Private" if epsilon is None else epsilon})
    write_rows(output / "released_update_manifest.csv", cache_rows)
    metrics = {
        "VictimRunID": run_id, "Method": method, "Epsilon": "INF / Non-Private" if epsilon is None else epsilon,
        "MAE": test["MAE"], "RMSE": test["RMSE"], "MAPE": test["MAPE"],
        "Val_MAE": validation["MAE"], "Val_RMSE": validation["RMSE"], "Val_MAPE": validation["MAPE"],
        "FinalTrainLoss": losses[-1], "AchievedEpsilon": calibration.get("achieved_epsilon", calibration.get("AchievedEpsilon", epsilon)),
        "ReleaseCount": len(collector.records), "ReleasedUpdateHash": cache_hash,
        "RuntimeSeconds": time.perf_counter() - started,
    }
    write_rows(output / "metrics.csv", [metrics])
    privacy = {
        "VictimRunID": run_id, "Method": method, "TargetEpsilon": metrics["Epsilon"],
        "AchievedEpsilon": metrics["AchievedEpsilon"], "Delta": "N/A" if epsilon is None else DELTA,
        "Adjacency": "N/A" if epsilon is None else "replacement", "ReleasesPerClient": RELEASES_PER_CLIENT,
        "SubsamplingAmplification": "NO", "NoiseSeedPolicyHash": policy_hash,
    }
    write_rows(output / "privacy_summary.csv", [privacy])
    utility_payload = {k: metrics[k] for k in ("VictimRunID", "Method", "Epsilon", "MAE", "RMSE", "MAPE", "AchievedEpsilon", "ReleaseCount", "ReleasedUpdateHash")}
    utility_hash = canonical_hash(utility_payload)
    (output / "utility_metrics_hash.txt").write_text(utility_hash + "\n", encoding="utf-8")
    (output / "environment.txt").write_text(_environment_snapshot(device), encoding="utf-8")
    return {
        "key": key, "run_id": run_id, "method": method, "epsilon": epsilon, "metrics": metrics,
        "records": by_client, "released_hash": cache_hash, "utility_hash": utility_hash,
        "config_hash": sha256_file(output / "config.yaml"), "core_config_hash": core_config_hash, "output": output,
    }


def compare_reproducibility(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    passed = a["released_hash"] == b["released_hash"] and a["core_config_hash"] == b["core_config_hash"]
    for metric in ("MAE", "RMSE", "MAPE"):
        left, right = float(a["metrics"][metric]), float(b["metrics"][metric])
        relative = abs(left - right) / max(abs(left), 1e-15)
        result[f"RunA_{metric}"] = left
        result[f"RunB_{metric}"] = right
        result[f"RelativeDeviation_{metric}"] = relative
        passed = passed and relative <= 1e-6
    result.update({
        "ReleasedUpdateHashA": a["released_hash"], "ReleasedUpdateHashB": b["released_hash"],
        "ReleasedUpdateHashExactMatch": a["released_hash"] == b["released_hash"],
        "CoreConfigHashA": a["core_config_hash"], "CoreConfigHashB": b["core_config_hash"],
        "CoreConfigExactMatch": a["core_config_hash"] == b["core_config_hash"],
        "ExactReproducibility": passed,
    })
    return result


def _attack_record(record: UpdateRecord) -> UpdateRecord:
    if not set(ALLOWED_LAYERS).issubset(record.layers):
        raise RuntimeError("released cache lacks locked attack layers")
    return UpdateRecord(record.round, record.client_id, {name: record.layers[name] for name in ALLOWED_LAYERS}, record.update_norm)


def _write_features(path: Path, features: Mapping[int, Mapping[str, np.ndarray]], run_id: str) -> None:
    rows = []
    for client_id in range(CLIENTS):
        row: dict[str, Any] = {"VictimRunID": run_id, "ClientID": client_id}
        for family in ("norm_stats", "cross_round_stats", "proposed_spectral"):
            for index, value in enumerate(features[client_id][family]):
                row[f"{family}_{index:03d}"] = float(value)
        rows.append(row)
    write_rows(path, rows)


def run_attacks(victims: Sequence[Mapping[str, Any]], attack_root: Path, property_info: Mapping[str, Any], folds: Mapping[str, np.ndarray], feature_hash: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    attack_root.mkdir(parents=True, exist_ok=False)
    fold_dir = attack_root / "fold_assignments"
    fold_dir.mkdir(parents=True, exist_ok=False)
    for property_name in PROPERTIES:
        write_rows(fold_dir / f"{PROPERTY_DISPLAY[property_name].replace(' ', '_')}_folds.csv", _property_fold_rows(property_name, property_info[property_name], folds[property_name]))
    all_rows: list[dict[str, Any]] = []
    hashes: dict[str, dict[str, str]] = {}
    for victim in victims:
        key, run_id = str(victim["key"]), str(victim["run_id"])
        output = attack_root / key
        output.mkdir(parents=True, exist_ok=False)
        features = {
            client_id: extract_feature_families([_attack_record(row) for row in victim["records"][client_id]])
            for client_id in range(CLIENTS)
        }
        feature_path = output / "attack_features.csv"
        _write_features(feature_path, features, run_id)
        prediction_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        bootstrap_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        for property_name in PROPERTIES:
            label_rows.extend({"VictimRunID": run_id, **row} for row in _property_fold_rows(property_name, property_info[property_name], folds[property_name]))
            labels = property_info[property_name]["labels"]
            for attack_name in ATTACKS:
                x = None if attack_name == "random_guess" else np.stack([features[i][attack_name] for i in range(CLIENTS)])
                display = "FedAvg R2" if victim["epsilon"] is None else f"{victim['method']} R2 eps{victim['epsilon']}"
                metric, predictions, bootstrap = evaluate_attack(display, property_name, attack_name, x, labels, folds[property_name])
                metric.update({"VictimRunID": run_id, "Method": victim["method"], "Epsilon": "INF / Non-Private" if victim["epsilon"] is None else victim["epsilon"]})
                metric_rows.append(metric)
                prediction_rows.extend({"VictimRunID": run_id, **row} for row in predictions)
                bootstrap_rows.append({"VictimRunID": run_id, **bootstrap})
        write_rows(output / "property_labels.csv", label_rows)
        write_rows(output / "oof_predictions.csv", prediction_rows)
        write_rows(output / "attack_metrics.csv", metric_rows)
        write_rows(output / "bootstrap_ci.csv", bootstrap_rows)
        save_config(output / "attack_config.yaml", {
            "VictimRunID": run_id, "feature_spec_sha256": feature_hash,
            "classifier": "L2 Logistic Regression", "C": 1.0, "attack_cv_seed": ATTACK_CV_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "properties": list(PROPERTIES), "peak_pattern_evaluated": False,
            "observation": "same-run server-visible released updates only",
        })
        hashes[key] = {"feature": sha256_file(feature_path), "metrics": sha256_file(output / "attack_metrics.csv")}
        all_rows.extend(metric_rows)
    return all_rows, hashes


def _proposed(all_rows: Sequence[Mapping[str, Any]], run_id: str, property_name: str) -> Mapping[str, Any]:
    display = PROPERTY_DISPLAY[property_name]
    return next(row for row in all_rows if row["VictimRunID"] == run_id and row["Property"] == display and row["AttackFeature"] == "proposed_spectral")


def write_report(path: Path, policy_hash: str, repro: Mapping[str, Any], victims: Sequence[Mapping[str, Any]], attacks: Sequence[Mapping[str, Any]], historical_frontier: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    lines = [
        "# Locked-Seed Privacy-Utility Attack Report", "",
        "## 1-3. Why the old replay stopped", "",
        "The earlier representative attack replay stopped because it required a new stochastic DP trajectory to remain within 5% of historical MAE/RMSE. Historical DP runs used unrecorded `secrets.randbits(63)` seeds and retained no server-visible released-update cache, so exact recovery is impossible. Historical results were not modified and remain development references only.", "",
        "## 4-7. Locked seed policy and reproducibility", "",
        f"The immutable policy uses base seed `{BASE_DP_NOISE_SEED}` and SHA-256 domain separation over method, epsilon, round, client, group, and tensor. Policy hash: `{policy_hash}`.", "",
        f"Run A/B DP-FedAvg epsilon 2048 produced MAE {float(repro['RunA_MAE']):.10f}/{float(repro['RunB_MAE']):.10f}, RMSE {float(repro['RunA_RMSE']):.10f}/{float(repro['RunB_RMSE']):.10f}, and identical cache hashes: **{repro['ReleasedUpdateHashExactMatch']}**. Exact reproducibility: **{repro['ExactReproducibility']}**.", "",
        "## 8-9. Five self-consistent victims", "",
        "| VictimRunID | Method | Epsilon | MAE | RMSE | MAPE | Released update hash |", "|---|---|---:|---:|---:|---:|---|",
    ]
    for victim in victims:
        m = victim["metrics"]
        lines.append(f"| {victim['run_id']} | {victim['method']} | {m['Epsilon']} | {float(m['MAE']):.4f} | {float(m['RMSE']):.4f} | {float(m['MAPE']):.4f} | `{victim['released_hash']}` |")
    lines += ["", "Every utility metric, release cache, attack feature, prediction, and attack metric is linked by one VictimRunID.", "", "## 10-15. Attack protocol and results", "", "Only server-visible released updates are used. Extreme Event and Periodicity Strength use the pre-locked R2 feature specification; Peak Pattern is not evaluated. L2 logistic regression uses client-level stratified OOF folds. Raw AUC and direction-independent separability are reported with 2,000-replicate client bootstrap intervals.", "", "| Property | VictimRunID | Raw AUC | AUC separability | 95% CI | Accuracy | F1 |", "|---|---|---:|---:|---|---:|---:|" ]
    for row in attacks:
        if row["AttackFeature"] == "proposed_spectral":
            lines.append(f"| {row['Property']} | {row['VictimRunID']} | {float(row['ROCAUCRaw']):.4f} | {float(row['ROCAUCSeparability']):.4f} | [{float(row['AUCSeparabilityCI95Low']):.4f}, {float(row['AUCSeparabilityCI95High']):.4f}] | {float(row['Accuracy']):.4f} | {float(row['F1']):.4f} |")
    decision = decide(attacks)
    lines += [
        "", "## 16-18. Matched comparison and historical reference", "",
        f"PJSDP versus matched DP-FedAvg empirical leakage is {decision['comparison'][2048]} at epsilon 2048 and {decision['comparison'][1024]} at epsilon 1024. Formal epsilon and empirical property-inference performance remain distinct quantities; attack AUC is not a DP proof.", "",
        f"Historical development results are retained at `{historical_frontier}` but are not paired with these locked-seed attack results.", "",
        "## 19. Self-consistency", "", "**SELF-CONSISTENT EVIDENCE = YES.**", "",
        "## 20. Only one next recommendation", "", f"**{decision['next']}**", "",
        f"Overall conclusion: **{decision['conclusion']}**.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_results_namespace(results_root: Path) -> None:
    """Allow immutable policy and archived failed attempts, but never active outputs."""
    if not results_root.exists():
        return
    forbidden = [
        path.name
        for path in results_root.iterdir()
        if path.name != "locked_noise_seed_policy.json"
        and not (path.is_dir() and path.name.startswith("seed42_failed_"))
    ]
    if forbidden:
        raise FileExistsError(
            f"refusing to overwrite non-policy content in {results_root}: {sorted(forbidden)}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_results_namespace(args.results_root)
    args.results_root.mkdir(parents=True, exist_ok=True)
    policy_hash = sha256_file(args.noise_policy)
    if policy_hash != EXPECTED_POLICY_SHA256:
        raise ValueError(f"locked policy hash mismatch: {policy_hash}")
    policy = json.loads(args.noise_policy.read_text(encoding="utf-8"))
    if policy["base_dp_noise_seed"] != BASE_DP_NOISE_SEED or policy["status"] != "LOCKED_BEFORE_EVALUATION":
        raise ValueError("locked noise policy contents changed")
    root_policy = args.results_root / "locked_noise_seed_policy.json"
    if not root_policy.exists():
        root_policy.write_bytes(args.noise_policy.read_bytes())
    elif sha256_file(root_policy) != policy_hash:
        raise ValueError("results namespace contains a different seed policy")
    feature_spec, feature_hash = validate_feature_spec(args.feature_spec)
    if feature_hash != EXPECTED_FEATURE_SPEC_SHA256:
        raise ValueError("feature specification changed")
    schedule, _ = load_and_verify_schedule(args.schedule_path)
    schedule_hash = sha256_file(args.schedule_path)
    if schedule_hash != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("schedule changed")
    reference = _load_reference_config(args.reference_config)
    expected = {"dataset": "pems08", "seed": 42, "num_clients": 50, "clients_per_round": 5, "local_epochs": 1}
    for field, value in expected.items():
        if reference.get(field) != value:
            raise ValueError(f"reference config mismatch: {field}")
    if args.report_path.exists():
        raise FileExistsError(args.report_path)
    seed_root = args.results_root / "seed42"
    seed_root.mkdir(parents=True, exist_ok=False)
    if args.legacy_replay.exists():
        marker = args.legacy_replay / "LEGACY_NON_REPRODUCIBLE_NOISE_TRAJECTORY.txt"
        if not marker.exists():
            marker.write_text("LEGACY / NON-REPRODUCIBLE-NOISE-TRAJECTORY\nExcluded from locked-seed evidence.\n", encoding="utf-8")
    prepared = _prepare_data(reference, args.data_root, "pems08")
    device = _resolve_device(args.device)
    repro_root = seed_root / "reproducibility_check"
    repro_root.mkdir(parents=True, exist_ok=False)
    run_a = run_victim(key="repro_a", method="DP-FedAvg", epsilon=2048, output=repro_root / "run_a", reference=reference, prepared=prepared, schedule=schedule, device=device, policy_hash=policy_hash)
    run_b = run_victim(key="repro_b", method="DP-FedAvg", epsilon=2048, output=repro_root / "run_b", reference=reference, prepared=prepared, schedule=schedule, device=device, policy_hash=policy_hash)
    repro = compare_reproducibility(run_a, run_b)
    write_rows(repro_root / "reproducibility_check.csv", [repro])
    if not repro["ExactReproducibility"]:
        print("Exact reproducibility:\nNO\nSTOP")
        return {"reproducibility": repro, "stopped": True}
    victims = [
        run_victim(key=key, method=method, epsilon=epsilon, output=seed_root / key, reference=reference, prepared=prepared, schedule=schedule, device=device, policy_hash=policy_hash)
        for key, method, epsilon in VICTIMS
    ]
    property_info = build_property_labels(args.data_root, feature_spec)
    folds = {name: make_folds(info["labels"], int(info["n_splits"])) for name, info in property_info.items()}
    attack_rows, attack_hashes = run_attacks(victims, seed_root / "attack_results", property_info, folds, feature_hash)
    utility_table = []
    for victim in victims:
        m = victim["metrics"]
        utility_table.append({k: m[k] for k in ("VictimRunID", "Method", "Epsilon", "MAE", "RMSE", "MAPE", "AchievedEpsilon", "ReleaseCount", "ReleasedUpdateHash")})
    write_rows(seed_root / "locked_seed_utility_table.csv", utility_table)
    requested_attack_rows = [{
        "VictimRunID": row["VictimRunID"], "Property": row["Property"], "Attack": row["AttackFeature"],
        "Accuracy": row["Accuracy"], "F1": row["F1"], "RawAUC": row["ROCAUCRaw"],
        "AUCSeparability": row["ROCAUCSeparability"], "CI95Low": row["AUCSeparabilityCI95Low"],
        "CI95High": row["AUCSeparabilityCI95High"],
    } for row in attack_rows]
    write_rows(seed_root / "locked_seed_attack_table.csv", requested_attack_rows)
    combined = []
    for victim in victims:
        extreme = _proposed(attack_rows, victim["run_id"], "extreme_event")
        periodic = _proposed(attack_rows, victim["run_id"], "periodicity_strength")
        combined.append({
            "VictimRunID": victim["run_id"], "Method": victim["method"],
            "Epsilon": "INF / Non-Private" if victim["epsilon"] is None else victim["epsilon"],
            "MAE": victim["metrics"]["MAE"], "RMSE": victim["metrics"]["RMSE"],
            "ExtremeAUCsep": extreme["ROCAUCSeparability"], "PeriodicityAUCsep": periodic["ROCAUCSeparability"],
        })
    write_rows(seed_root / "locked_seed_combined_table.csv", combined)
    manifest_rows = []
    for victim in victims:
        hashes = attack_hashes[victim["key"]]
        manifest_rows.append({
            "VictimRunID": victim["run_id"], "Method": victim["method"],
            "Epsilon": "INF / Non-Private" if victim["epsilon"] is None else victim["epsilon"],
            "Delta": "N/A" if victim["epsilon"] is None else DELTA, "Seed": 42,
            "ScheduleHash": EXPECTED_SCHEDULE_SHA256, "ConfigHash": victim["config_hash"],
            "NoiseSeedPolicyHash": policy_hash, "ReleasedUpdateHash": victim["released_hash"],
            "UtilityMetricsHash": victim["utility_hash"], "AttackFeatureHash": hashes["feature"],
            "AttackMetricsHash": hashes["metrics"], "SelfConsistent": True,
        })
    write_rows(args.results_root / "evidence_manifest.csv", manifest_rows)
    write_report(args.report_path, policy_hash, repro, victims, attack_rows, args.historical_frontier)
    decision = decide(attack_rows)
    print("========================================\nLOCKED-SEED SELF-CONSISTENT EVALUATION\n======================================")
    print(f"\nHistorical DP trajectory reproducible:\nNO\n\nHistorical results modified:\nNO\n\nNew locked DP seed policy:\nENABLED\n\nNoise seed policy hash:\n{policy_hash}")
    print(f"\nReproducibility check:\n\nRun A MAE = {repro['RunA_MAE']}\nRun B MAE = {repro['RunB_MAE']}\n\nRun A RMSE = {repro['RunA_RMSE']}\nRun B RMSE = {repro['RunB_RMSE']}\n\nReleased update hash A = {repro['ReleasedUpdateHashA']}\nReleased update hash B = {repro['ReleasedUpdateHashB']}\n\nExact reproducibility:\nYES")
    print("\n---\n\n## NEW LOCKED-SEED UTILITY")
    for row in utility_table:
        print(f"\n{row['VictimRunID']}:\nMAE = {row['MAE']}\nRMSE = {row['RMSE']}\nMAPE = {row['MAPE']}")
    print("\n---\n\n## NEW LOCKED-SEED ATTACK")
    for property_name in PROPERTIES:
        print(f"\n{PROPERTY_DISPLAY[property_name]}:")
        for victim in victims:
            row = _proposed(attack_rows, victim["run_id"], property_name)
            print(f"{victim['run_id']}: Raw AUC={float(row['ROCAUCRaw']):.6f}, AUC Sep={float(row['ROCAUCSeparability']):.6f}, CI=[{float(row['AUCSeparabilityCI95Low']):.6f}, {float(row['AUCSeparabilityCI95High']):.6f}]")
    print(f"\n---\n\nSELF-CONSISTENT EVIDENCE:\nYES\n\nHistorical development results retained:\nYES\n\nHistorical utility paired with new attack:\nNO\n\nFinal conclusion:\n{decision['conclusion']}\n\nNext recommended experiment:\n{decision['next']}")
    return {"reproducibility": repro, "victims": victims, "attacks": attack_rows, "decision": decision}


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
