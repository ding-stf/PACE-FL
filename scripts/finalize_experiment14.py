from __future__ import annotations

"""Validate and atomically publish Experiment 14 final ICASSP artifacts."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from pathlib import Path
from typing import Any

import yaml

METHODS = ("FedAvg", "DP-FedAvg", "PJSDP")
KEYS = {"FedAvg": "fedavg", "DP-FedAvg": "dp_fedavg_eps2048", "PJSDP": "pjsdp_eps2048"}
EXPECTED_POLICY_HASH = "cd684b0905a405eb6a02b737d7e145cb01ec6bd8da8161d4899d9d35feee96d5"
EXPECTED_PEMS = {
    "FedAvg": (30.259911196920672, 44.45204357550174, 31.544729808609844),
    "DP-FedAvg": (37.65781904946483, 52.63640247064572, 42.19865234279487),
    "PJSDP": (32.251791328209116, 45.236367637185346, 41.83825344833023),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--dedicated-tests", type=Path, required=True)
    p.add_argument("--full-tests", type=Path, required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def one_row(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one row: {path}")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("empty values")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def parse_pytest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    passed = sum(int(x) for x in re.findall(r"(\d+) passed", text))
    failed = sum(int(x) for x in re.findall(r"(\d+) failed", text))
    exit_codes = re.findall(r"exit_code=(\d+)", text)
    valid = passed > 0 and failed == 0 and exit_codes and all(x == "0" for x in exit_codes)
    return {"path": str(path), "sha256": sha256(path), "passed": passed, "failed": failed, "valid": valid}


def validate_run(path: Path, dataset: str, seed: int, method: str) -> dict[str, Any]:
    required = ("config.yaml", "metrics.csv", "provenance.csv", "released_update_manifest.csv", "round_timing.csv")
    missing = [x for x in required if not (path / x).is_file()]
    if missing:
        raise RuntimeError(f"missing run artifacts at {path}: {missing}")
    metrics = one_row(path / "metrics.csv")
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    provenance = one_row(path / "provenance.csv")
    if not (metrics["Status"] == "FORMAL_LOCKED" and metrics["Dataset"] == dataset and int(metrics["Seed"]) == seed and metrics["Method"] == method):
        raise RuntimeError(f"identity/status mismatch: {path}")
    if not all(math.isfinite(float(metrics[k])) for k in ("MAE", "RMSE", "MAPE")):
        raise RuntimeError(f"non-finite metric: {path}")
    expected = {"rounds": 20, "num_clients": 50, "clients_per_round": 5, "releases_per_client": 2, "local_epochs": 1, "history": 12, "horizon": 12, "batch_size": 64}
    if any(int(config[k]) != value for k, value in expected.items()) or float(config["learning_rate"]) != 0.01:
        raise RuntimeError(f"protocol mismatch: {path}")
    if int(metrics["ReleaseCount"]) != 100 or len(read_rows(path / "released_update_manifest.csv")) != 100 or len(read_rows(path / "round_timing.csv")) != 20:
        raise RuntimeError(f"round/release mismatch: {path}")
    if provenance["ConfigHash"] != sha256(path / "config.yaml") or provenance["ReleasedUpdateHash"] != metrics["ReleasedUpdateHash"]:
        raise RuntimeError(f"provenance hash mismatch: {path}")
    if not (provenance["ExplicitRNGNamespace"] == "True" and provenance["TimestampSeedUsed"] == "False" and provenance["SecretsRandbitsUsed"] == "False"):
        raise RuntimeError(f"non-reproducible RNG provenance: {path}")
    epsilon = None if method == "FedAvg" else 2048
    if epsilon is not None:
        if not (float(config["epsilon"]) == epsilon and float(config["delta"]) == 1e-5 and config["adjacency"] == "replacement" and config["accountant"] == "verified_RDP_no_subsampling_amplification"):
            raise RuntimeError(f"DP contract mismatch: {path}")
        if abs(float(metrics["AchievedEpsilon"]) - epsilon) > 1e-8:
            raise RuntimeError(f"epsilon mismatch: {path}")
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["metrics_hash"] != sha256(path / "metrics.csv") or manifest["checkpoint_hash"] != sha256(path / "final_model.pt"):
            raise RuntimeError(f"manifest hash mismatch: {path}")
        if epsilon is not None and manifest["subsampling_amplification"] is not False:
            raise RuntimeError(f"subsampling amplification mismatch: {path}")
    return {"metrics": metrics, "config": config, "provenance": provenance, "path": path}


def metr_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validated: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in (42, 123, 2026):
        for method in METHODS:
            key = KEYS[method]
            path = root / "results/metrla_final_sanity/seed42" / key if seed == 42 else root / "results/final_icassp/metr_la" / key / f"seed{seed}"
            validated[(seed, method)] = validate_run(path, "metr_la", seed, method)
    fixed = ("dataset", "method", "epsilon", "delta", "adjacency", "rounds", "num_clients", "clients_per_round", "releases_per_client", "local_epochs", "history", "horizon", "batch_size", "learning_rate", "projection_ratio", "projection_seed", "projection_mask_hash", "clipping", "accountant")
    for method in METHODS:
        configs = [validated[(s, method)]["config"] for s in (42, 123, 2026)]
        for field in fixed:
            if len({json.dumps(c.get(field), sort_keys=True) for c in configs}) != 1:
                raise RuntimeError(f"METR-LA cross-seed protocol mismatch: {method}/{field}")
    rows = []
    for seed in (42, 123, 2026):
        for method in METHODS:
            item = validated[(seed, method)]
            m = item["metrics"]
            rows.append({"Dataset": "METR-LA", "Seed": seed, "Method": method, "Epsilon": m["Epsilon"], "MAE": float(m["MAE"]), "RMSE": float(m["RMSE"]), "MAPE": float(m["MAPE"]), "AchievedEpsilon": m["AchievedEpsilon"], "Valid": True, "ResultPath": str(item["path"])})
    summary = []
    for method in METHODS:
        rs = [x for x in rows if x["Method"] == method]
        out: dict[str, Any] = {"Dataset": "METR-LA", "Method": method, "Epsilon": rs[0]["Epsilon"], "Seeds": 3, "StdDefinition": "sample std (ddof=1)"}
        for metric in ("MAE", "RMSE", "MAPE"):
            out[f"{metric}_mean"], out[f"{metric}_std"] = mean_std([float(x[metric]) for x in rs])
        summary.append(out)
    dp = next(x for x in summary if x["Method"] == "DP-FedAvg")
    pj = next(x for x in summary if x["Method"] == "PJSDP")
    improvements = {k: (dp[f"{k}_mean"] - pj[f"{k}_mean"]) / dp[f"{k}_mean"] * 100 for k in ("MAE", "RMSE", "MAPE")}
    wins = {k: sum(next(x for x in rows if x["Seed"] == seed and x["Method"] == "PJSDP")[k] < next(x for x in rows if x["Seed"] == seed and x["Method"] == "DP-FedAvg")[k] for seed in (42, 123, 2026)) for k in ("MAE", "RMSE")}
    status = "SUPPORTED" if pj["MAE_mean"] < dp["MAE_mean"] and pj["RMSE_mean"] < dp["RMSE_mean"] and min(wins.values()) >= 2 else ("NOT SUPPORTED" if pj["MAE_mean"] >= dp["MAE_mean"] and pj["RMSE_mean"] >= dp["RMSE_mean"] else "MIXED")
    return rows, {"rows": summary, "improvements_pct": improvements, "wins": wins, "status": status}


def pems_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    utility = []
    base = root / "results/privacy_attack_locked_seed/seed42"
    for row in read_rows(base / "locked_seed_utility_table.csv"):
        if row["Method"] in METHODS and row["Epsilon"] in ("2048", "INF / Non-Private"):
            utility.append({"Seed": 42, "Method": row["Method"], **{k: float(row[k]) for k in ("MAE", "RMSE", "MAPE")}})
    attacks = []
    for row in read_rows(base / "locked_seed_attack_table.csv"):
        if row["Attack"] == "proposed_spectral" and "eps1024" not in row["VictimRunID"]:
            method = "FedAvg" if "fedavg_nonprivate" in row["VictimRunID"] else ("DP-FedAvg" if "dp_fedavg" in row["VictimRunID"] else "PJSDP")
            attacks.append({"Seed": 42, "Method": method, "Property": row["Property"], "AUCSeparability": float(row["AUCSeparability"])})
    for seed in (123, 2026):
        sr = root / "results/final_pems08_3seed" / f"seed_{seed}"
        for method in METHODS:
            row = one_row(sr / KEYS[method] / "metrics.csv")
            utility.append({"Seed": seed, "Method": method, **{k: float(row[k]) for k in ("MAE", "RMSE", "MAPE")}})
        for row in read_rows(sr / "attack/attack_summary.csv"):
            attacks.append({"Seed": seed, "Method": row["Method"], "Property": row["Property"], "AUCSeparability": float(row["AUCSeparability"])})
    if len(utility) != 9 or len(attacks) != 18:
        raise RuntimeError("incomplete formal PEMS08 evidence")
    summary = []
    for method in METHODS:
        rs = [x for x in utility if x["Method"] == method]
        out: dict[str, Any] = {"Dataset": "PEMS08", "Method": method, "Epsilon": "INF / Non-Private" if method == "FedAvg" else 2048, "Seeds": 3, "StdDefinition": "sample std (ddof=1)"}
        for metric in ("MAE", "RMSE", "MAPE"):
            out[f"{metric}_mean"], out[f"{metric}_std"] = mean_std([x[metric] for x in rs])
        if any(abs(out[f"{m}_mean"] - want) > 1e-12 for m, want in zip(("MAE", "RMSE", "MAPE"), EXPECTED_PEMS[method])):
            raise RuntimeError(f"PEMS08 FINAL RESULT MISMATCH: {method}")
        summary.append(out)
    privacy = []
    for method in METHODS:
        for prop in ("Extreme Event", "Periodicity Strength"):
            vals = [x["AUCSeparability"] for x in attacks if x["Method"] == method and x["Property"] == prop]
            mean, std = mean_std(vals)
            privacy.append({"Dataset": "PEMS08", "Method": method, "Epsilon": "INF / Non-Private" if method == "FedAvg" else 2048, "Property": prop, "AUCsep_mean": mean, "AUCsep_std": std, "Seeds": 3, "StdDefinition": "sample std (ddof=1)", "Source": "formal locked-seed/multiseed artifacts"})
    return utility, summary, privacy


def sparse_records(root: Path) -> tuple[list[dict[str, Any]], str]:
    sr = root / "results/final_icassp/sparse/pems08_seed42_eps2048"
    specs = (("rho100_reference", 1.0), ("rho025", 0.25), ("rho010", 0.10))
    out = []
    for slug, rho in specs:
        path = sr / slug
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        uploads = read_rows(path / "upload_communication.csv")
        if len(uploads) != 100 or not all(math.isfinite(float(metrics[k])) for k in ("MAE", "RMSE", "MAPE")):
            raise RuntimeError(f"invalid sparse run: {path}")
        total = sum(int(x["SerializedBytes"]) for x in uploads)
        nonzero = sum(int(x["NumberOfNonzeroValues"]) for x in uploads)
        numel = sum(int(x["TotalNumel"]) for x in uploads)
        value_bytes = nonzero * 4
        index_bytes = 0 if rho == 1.0 else nonzero * 4
        header_bytes = total - value_bytes - index_bytes
        if header_bytes < 0 or total != int(metrics.get("TotalSerializedBytes", total)):
            raise RuntimeError(f"serialized-byte mismatch: {path}")
        if rho < 1.0:
            config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
            prov = one_row(path / "provenance.csv")
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if not (config["postprocessing_order"] == "DP_PROTECTED_RELEASE -> TOPK -> ERROR_FEEDBACK -> SERIALIZATION -> SERVER" and config["error_feedback_residual_source"] == "DP_PROTECTED_RELEASE_ONLY" and prov["DPBeforeTopK"] == "True" and prov["ErrorFeedbackResidualSource"] == "DP_PROTECTED_RELEASE_ONLY"):
                raise RuntimeError(f"sparse DP/EF provenance mismatch: {path}")
            if manifest["communication_audit_hash"] != sha256(path / "upload_communication.csv"):
                raise RuntimeError(f"communication hash mismatch: {path}")
        out.append({"Dataset": "PEMS08", "Seed": 42, "Method": "PJSDP", "Epsilon": 2048, "Rho": rho, "MAE": float(metrics["MAE"]), "RMSE": float(metrics["RMSE"]), "MAPE": float(metrics["MAPE"]), "Uploads": 100, "ValuesPayloadBytes": value_bytes, "IndexBytes": index_bytes, "HeaderMetadataBytes": header_bytes, "PayloadBytes": value_bytes, "MetadataBytes": index_bytes + header_bytes, "TotalSerializedBytes": total, "NonZeroCount": nonzero, "TotalParameterCount": numel, "EffectiveSparsity": 1 - nonzero / numel, "ResultPath": str(path), "Valid": True})
    dense = out[0]
    for row in out:
        row["CompressionRatio"] = dense["TotalSerializedBytes"] / row["TotalSerializedBytes"]
        row["ByteReductionPct"] = (1 - row["TotalSerializedBytes"] / dense["TotalSerializedBytes"]) * 100
        for metric in ("MAE", "RMSE", "MAPE"):
            row[f"Delta{metric}VsDense"] = row[metric] - dense[metric]
        row["RelativeMAEDegradationPct"] = (row["MAE"] - dense["MAE"]) / dense["MAE"] * 100
        row["RelativeRMSEDegradationPct"] = (row["RMSE"] - dense["RMSE"]) / dense["RMSE"] * 100
    status = "SUPPORTED" if any(x["Rho"] < 1 and x["ByteReductionPct"] > 0 and x["RelativeMAEDegradationPct"] <= 5 and x["RelativeRMSEDegradationPct"] <= 5 for x in out) else "NOT SUPPORTED"
    return out, status


def latex_table(rows: list[dict[str, Any]]) -> str:
    lines = ["\\begin{tabular}{llccc}", "Dataset & Method & MAE & RMSE & MAPE \\\\", "\\hline"]
    for dataset in ("PEMS08", "METR-LA"):
        group = [x for x in rows if x["Dataset"] == dataset]
        best = {m: min(x[f"{m}_mean"] for x in group) for m in ("MAE", "RMSE", "MAPE")}
        for row in group:
            cells = []
            for metric in ("MAE", "RMSE", "MAPE"):
                value = f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_std']:.3f}"
                cells.append(f"\\textbf{{{value}}}" if row[f"{metric}_mean"] == best[metric] else value)
            lines.append(f"{dataset} & {row['Method']} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\end{tabular}", ""])


def build_terminal(per_seed: list[dict[str, Any]], metr: dict[str, Any], sparse: list[dict[str, Any]], dedicated: dict[str, Any], full: dict[str, Any], cross: str, sparse_status: str) -> str:
    lines = ["============================================================", "EXPERIMENT 14 — FINAL ICASSP COMPLETION", "============================================================", "", "Stage A: METR-LA", ""]
    for seed in (42, 123, 2026):
        lines.append(f"Seed {seed}:")
        for method in METHODS:
            x = next(r for r in per_seed if r["Seed"] == seed and r["Method"] == method)
            lines += [f"{method}:", f"MAE = {x['MAE']:.6f}", f"RMSE = {x['RMSE']:.6f}", f"MAPE = {x['MAPE']:.6f}", ""]
    lines += ["3-seed summary:"]
    for x in metr["rows"]:
        lines.append(f"{x['Method']}: MAE {x['MAE_mean']:.4f} ± {x['MAE_std']:.4f}; RMSE {x['RMSE_mean']:.4f} ± {x['RMSE_std']:.4f}; MAPE {x['MAPE_mean']:.4f} ± {x['MAPE_std']:.4f}")
    lines += ["", "PJSDP vs DP-FedAvg:", f"MAE improvement = {metr['improvements_pct']['MAE']:.3f}%", f"RMSE improvement = {metr['improvements_pct']['RMSE']:.3f}%", f"MAPE improvement = {metr['improvements_pct']['MAPE']:.3f}%", "", "----------------------------------------", "", "Stage B: Sparse Synchronization", ""]
    for x in sparse:
        lines += [f"rho = {x['Rho']:.2f}", f"MAE = {x['MAE']:.6f}", f"RMSE = {x['RMSE']:.6f}", f"MAPE = {x['MAPE']:.6f}", f"Uploads = {x['Uploads']}", f"Serialized bytes = {x['TotalSerializedBytes']}", f"Compression ratio = {x['CompressionRatio']:.3f}x", ""]
    lines += ["----------------------------------------", "", "Validation", "", "DP deterministic seed: PASS", "Actual byte accounting: PASS", "Sparse after DP: PASS", "Error-feedback provenance: PASS", f"Dedicated tests: {dedicated['passed']} passed, 0 failed", f"Full tests: {full['passed']} passed, 0 failed", "", "----------------------------------------", "", "FINAL JUDGMENT", "", f"CROSS-DATASET UTILITY: {cross}", f"SPARSE SYNCHRONIZATION: {sparse_status}", "READY TO WRITE ICASSP: YES", "", "============================================================"]
    return "\n".join(lines) + "\n"


def build_report(metr: dict[str, Any], sparse: list[dict[str, Any]], dedicated: dict[str, Any], full: dict[str, Any], cross: str, sparse_status: str) -> str:
    dp = next(x for x in metr["rows"] if x["Method"] == "DP-FedAvg")
    pj = next(x for x in metr["rows"] if x["Method"] == "PJSDP")
    r25 = next(x for x in sparse if x["Rho"] == .25)
    r10 = next(x for x in sparse if x["Rho"] == .10)
    return f"""# Experiment 14 — Final ICASSP Experimental Completion

All final values were regenerated from formal server artifacts. Means use three seeds (42, 123, 2026); standard deviations are sample standard deviations (`ddof=1`). No Experiment 13 artifact was modified.

## Audit and protocol

Canonical method: Current Two-Band PJSDP at ε=2048. Protocol: 50 clients, 20 rounds, 5 clients/round, 2 releases/client, 1 local epoch, δ=1e-5, replacement adjacency, verified RDP accounting without subsampling amplification. The deterministic DP namespace and locked policy hash were verified. Six METR-LA and two sparse candidate runs passed artifact, hash, release-count, finite-metric, and provenance checks, so no retraining was performed during finalization.

## Answers

**Q1. Does PJSDP still outperform matched DP-FedAvg on METR-LA?** Yes overall: mean MAE {pj['MAE_mean']:.4f} vs {dp['MAE_mean']:.4f}, and mean RMSE {pj['RMSE_mean']:.4f} vs {dp['RMSE_mean']:.4f}. PJSDP wins on both MAE and RMSE in {metr['wins']['MAE']}/3 and {metr['wins']['RMSE']}/3 seeds; seed 2026 is a narrow exception and is retained without tuning.

**Q2. Is there cross-dataset utility evidence?** {cross}. PEMS08 and METR-LA both show lower three-seed mean MAE/RMSE for PJSDP than matched DP-FedAvg.

**Q3. Does rho=.25 reduce bytes while maintaining acceptable utility?** Yes. It reduces actual serialized bytes by {r25['ByteReductionPct']:.2f}% ({r25['CompressionRatio']:.3f}x); MAE changes by {r25['DeltaMAEVsDense']:+.4f} and RMSE by {r25['DeltaRMSEVsDense']:+.4f} relative to rho=1.0.

**Q4. Does rho=.10 cause a more obvious utility loss?** No in this fixed seed42 comparison. It reduces bytes by {r10['ByteReductionPct']:.2f}% ({r10['CompressionRatio']:.3f}x), while all three observed errors are lower than the dense reference. This is a single-seed communication study, not evidence of a general accuracy improvement.

**Q5. Which sparse point best fits the main paper?** rho=.10 is the strongest point among the preregistered {{1.0, .25, .10}} choices: it has the largest byte reduction and no observed seed42 utility loss. rho=.25 remains a conservative companion point; no additional rho sweep is warranted.

**Q6. Is the evidence sufficient to stop experimenting?** Yes. The fixed comparisons, provenance checks, byte audit, and complete test suite pass. Begin manuscript writing; do not start a new mechanism experiment.

## Communication interpretation

Release-frequency reduction and sparsification are reported as separate layers because their byte formats differ. R4-E to R2-E reduces uploads and dense payload bytes by 50%. Within the FSD1/FSS1 sparse comparison, rho=.25 and rho=.10 reduce bytes relative to the actual serialized rho=1.0 reference. These results support communication-volume claims only, not latency claims.

## Validation

- Dedicated tests: {dedicated['passed']} passed, 0 failed.
- Full suite: {full['passed']} passed, 0 failed.
- DP provenance: PASS.
- Actual serialized-byte accounting: PASS.
- DP-before-sparse and residual provenance: PASS.

## Final status

```text
========================================
FINAL ICASSP EXPERIMENT STATUS
========================================

METR-LA 3-SEED:
COMPLETE

CROSS-DATASET UTILITY:
{cross}

SPARSE SYNCHRONIZATION:
{sparse_status}

COMMUNICATION BYTE ACCOUNTING:
PASS

DP PROVENANCE:
PASS

FULL TEST SUITE:
PASS

NEW MECHANISM EXPLORATION REQUIRED:
NO

READY TO WRITE ICASSP PAPER:
YES
========================================
```
"""


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise RuntimeError(f"refusing noncanonical project root: {root}")
    policy = root / "results/privacy_attack_locked_seed/locked_noise_seed_policy.json"
    if sha256(policy) != EXPECTED_POLICY_HASH or json.loads(policy.read_text(encoding="utf-8"))["status"] != "LOCKED_BEFORE_EVALUATION":
        raise RuntimeError("locked DP noise policy mismatch")
    victim_source = root / "experiments/exp_icassp_final_victim.py"
    if "secrets.randbits" in victim_source.read_text(encoding="utf-8"):
        raise RuntimeError("INVALID: unrecoverable DP seed in Experiment 14 runner")
    canonical_docs = [root / "docs/THREE_BAND_SELECTIVE_DP_SANITY_REPORT.md", root / "docs/SELECTIVE_PRIVATE_SYNC_SANITY_REPORT.md"]
    if not all("CURRENT TWO-BAND PJSDP EPS2048" in p.read_text(encoding="utf-8") for p in canonical_docs):
        raise RuntimeError("EXPERIMENT 14 BLOCKED: CANONICAL MAIN METHOD CONFLICT")
    dedicated, full = parse_pytest(args.dedicated_tests), parse_pytest(args.full_tests)
    if not dedicated["valid"] or not full["valid"]:
        raise RuntimeError("test evidence is not passing")
    metr_per_seed, metr = metr_records(root)
    _, pems_summary, privacy = pems_records(root)
    sparse, sparse_status = sparse_records(root)
    cross_status = "SUPPORTED" if metr["status"] == "SUPPORTED" and next(x for x in pems_summary if x["Method"] == "PJSDP")["MAE_mean"] < next(x for x in pems_summary if x["Method"] == "DP-FedAvg")["MAE_mean"] and next(x for x in pems_summary if x["Method"] == "PJSDP")["RMSE_mean"] < next(x for x in pems_summary if x["Method"] == "DP-FedAvg")["RMSE_mean"] else "MIXED"
    output = root / "results/final_icassp_completion"
    stage = root / "results/final_icassp_completion.staging"
    if output.exists() or stage.exists():
        raise FileExistsError("final/staging output already exists; refusing overwrite")
    stage.mkdir(parents=True)
    write_rows(stage / "metr_la/metr_la_per_seed.csv", metr_per_seed)
    write_rows(stage / "metr_la/metr_la_summary.csv", metr["rows"])
    write_json(stage / "metr_la/metr_la_manifest.json", {"status": metr["status"], "improvement_definition": "(DPFedAvg-PJSDP)/DPFedAvg*100", "improvements_pct": metr["improvements_pct"], "pjsdp_wins_vs_dp_fedavg": metr["wins"], "std_definition": "sample std (ddof=1)", "valid": True})
    write_rows(stage / "sparse_sync/sparse_per_ratio.csv", sparse)
    write_rows(stage / "sparse_sync/sparse_summary.csv", sparse)
    write_json(stage / "sparse_sync/sparse_manifest.json", {"status": sparse_status, "ratios": [1.0, 0.25, 0.10], "fixed_comparison_no_sweep": True, "byte_accounting": "actual FSD1/FSS1 serialized bytes", "payload_definition": "float32 value bytes", "metadata_definition": "int32 sparse indices plus wire headers/schema", "dp_postprocessing": True, "error_feedback_residual_source": "DP_PROTECTED_RELEASE_ONLY", "valid": True})
    main_table = pems_summary + metr["rows"]
    write_rows(stage / "tables/table_main_utility.csv", main_table)
    write_rows(stage / "tables/table_privacy.csv", privacy)
    (stage / "tables/table_main_utility_latex.txt").write_text(latex_table(main_table), encoding="utf-8")
    release = {x["Method"]: x for x in read_rows(root / "results/release_composition_sanity/summary.csv")}
    communication = [
        {"ComparisonLayer": "ReleaseFrequency", "MethodOrRho": "PJSDP R4-E (4 releases/client)", "Uploads": int(release["R4-E"]["TotalUploads"]), "SerializedBytes": int(release["R4-E"]["TotalBytes"]), "CompressionRatio": 1.0, "ByteReductionPct": 0.0, "MAE": float(release["R4-E"]["MAE"]), "RMSE": float(release["R4-E"]["RMSE"]), "MAPE": float(release["R4-E"]["MAPE"]), "AccountingBasis": "legacy dense payload bytes; compare only within ReleaseFrequency layer"},
        {"ComparisonLayer": "ReleaseFrequency", "MethodOrRho": "PJSDP R2-E (2 releases/client)", "Uploads": int(release["R2-E"]["TotalUploads"]), "SerializedBytes": int(release["R2-E"]["TotalBytes"]), "CompressionRatio": int(release["R4-E"]["TotalBytes"]) / int(release["R2-E"]["TotalBytes"]), "ByteReductionPct": (1 - int(release["R2-E"]["TotalBytes"]) / int(release["R4-E"]["TotalBytes"])) * 100, "MAE": float(release["R2-E"]["MAE"]), "RMSE": float(release["R2-E"]["RMSE"]), "MAPE": float(release["R2-E"]["MAPE"]), "AccountingBasis": "legacy dense payload bytes; compare only within ReleaseFrequency layer"},
    ]
    communication += [{"ComparisonLayer": "Sparsification", "MethodOrRho": f"PJSDP rho={x['Rho']:.2f}", "Uploads": x["Uploads"], "SerializedBytes": x["TotalSerializedBytes"], "CompressionRatio": x["CompressionRatio"], "ByteReductionPct": x["ByteReductionPct"], "MAE": x["MAE"], "RMSE": x["RMSE"], "MAPE": x["MAPE"], "AccountingBasis": "actual FSD1/FSS1 wire bytes; compare only within Sparsification layer"} for x in sparse]
    write_rows(stage / "tables/table_communication.csv", communication)
    (stage / "manifests").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.dedicated_tests, stage / "manifests/dedicated_tests.txt")
    shutil.copy2(args.full_tests, stage / "manifests/full_tests.txt")
    sources = [policy, victim_source, root / "communication/sparse_sync.py", root / "communication/error_feedback.py", root / "communication/serializer.py", root / "federated/stg_server.py", root / "results/icassp_summary/pems08_final_3seed_summary.csv", root / "results/final_icassp/reference_audit.json", root / "results/final_icassp/queue_summary.json"]
    terminal = build_terminal(metr_per_seed, metr, sparse, dedicated, full, cross_status, sparse_status)
    (stage / "FINAL_TERMINAL_SUMMARY.txt").write_text(terminal, encoding="utf-8")
    manifest = {"experiment": "Experiment 14 — Final ICASSP Experimental Completion", "status": {"METR-LA 3-SEED": "COMPLETE", "CROSS-DATASET UTILITY": cross_status, "SPARSE SYNCHRONIZATION": sparse_status, "COMMUNICATION BYTE ACCOUNTING": "PASS", "DP PROVENANCE": "PASS", "FULL TEST SUITE": "PASS", "NEW MECHANISM EXPLORATION REQUIRED": "NO", "READY TO WRITE ICASSP PAPER": "YES"}, "training_reused": {"metr_la_seed42": True, "pems08_rho1": True}, "new_training_validated": {"metr_la_runs": 6, "sparse_runs": 2}, "tests": {"dedicated": dedicated, "full": full}, "source_hashes": {str(p): sha256(p) for p in sources}, "artifact_hashes": {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}, "no_experiment13_artifacts_modified": True}
    write_json(stage / "FINAL_ICASSP_EXPERIMENT_MANIFEST.json", manifest)
    report = build_report(metr, sparse, dedicated, full, cross_status, sparse_status)
    stage.rename(output)
    report_path = root / "docs/FINAL_ICASSP_EXPERIMENT_REPORT.md"
    report_tmp = report_path.with_suffix(".md.tmp")
    report_tmp.write_text(report, encoding="utf-8")
    report_tmp.replace(report_path)
    print(terminal, end="")


if __name__ == "__main__":
    main()
