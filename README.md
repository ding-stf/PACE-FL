# PACE-FL

Core code for **PACE-FL: Privacy-Aware Efficient Federated Traffic Forecasting with Sparse Model Release**.

## Main implementation

| Paper component | Source |
| --- | --- |
| DIM-SR: exact-dimensional orthogonal mixing | `communication/exact_dimension_orthogonal_mixing.py` |
| DIM-SR: implicit-mask wire encoding/decoding | `communication/implicit_mask_rsr.py` |
| Randomized sparse release | `communication/randomized_spectral_release.py` |
| TGP-OM: task-gradient history, protected subspace, calibration and projection | `experiments_revision/task_online_convergence_20260907/train.py` (`calibrate`, `project`, `aux_loss`) |
| Property-direction estimation | `experiments_revision/directional_projection_pilot_20260907/run_projection.py` (`fit_map`) |
| Privacy evaluation | `experiments_revision/task_online_convergence_20260907/attack.py`, `attacks/` |
| Forecasting model and federated training | `models/stgcn.py`, `federated/` |
| Traffic windows, partitioning and dynamic graphs | `datasets/` |

Internal names such as ED-OM-RSR, RSR and FedSpecShield predate the manuscript terminology. They are retained to preserve the original implementation and wire formats. Supporting experiment modules are included only where needed by the core code's imports.

## Installation and core checks

Use Python 3.10 or later and install a PyTorch build appropriate for your hardware.

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

## Data

Datasets are **not included**. Obtain METR-LA and PEMS08 separately under their respective terms. Data preparation code is provided:

```bash
python -m scripts.prepare_data --help
python -m scripts.download_traffic_data --help
```

The expected normalized files are `data/<dataset>/values.npy` and `adjacency.npy`, optionally `coordinates.npy`. See the adapters for tensor shapes and temporal splitting. The `datasets/` directory contains Python loaders, not dataset contents.

## Scope and reproducibility

This is a **core-source release**, not the full experiment archive. Datasets, checkpoints, observations, logs, manuscript files, figures, server scripts and experiment queues are excluded. Historical experiment runners reference frozen configurations, labels, schedules or intermediate records that are not bundled here; they are retained as implementation references and are not advertised as standalone commands reproducing the paper's tables. Four archive-only integration checks are explicitly skipped when their excluded artifacts are absent. The included unit tests exercise the core transformations, serialization and STGCN without those archives.

Algorithms are copied from A3. Host-specific project-root constants are made relative to the checkout. `SOURCE_MANIFEST.json` records original and released SHA-256 hashes for each source file. This release does not assign a new license.
