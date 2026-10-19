from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DCRNN_DRIVE_FILES = {
    "metr_la": (
        "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC",
        "metr-la.h5",
    ),
    "pems_bay": (
        "1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq",
        "pems-bay.h5",
    ),
}

DCRNN_GRAPH_BASE = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/"
    "master/data/sensor_graph"
)

HTTP_FILES = {
    "dcrnn/sensor_graph/adj_mx.pkl": f"{DCRNN_GRAPH_BASE}/adj_mx.pkl",
    "dcrnn/sensor_graph/adj_mx_bay.pkl": f"{DCRNN_GRAPH_BASE}/adj_mx_bay.pkl",
    "dcrnn/sensor_graph/distances_la_2012.csv": (
        f"{DCRNN_GRAPH_BASE}/distances_la_2012.csv"
    ),
    "dcrnn/sensor_graph/distances_bay_2017.csv": (
        f"{DCRNN_GRAPH_BASE}/distances_bay_2017.csv"
    ),
    "dcrnn/sensor_graph/graph_sensor_ids.txt": (
        f"{DCRNN_GRAPH_BASE}/graph_sensor_ids.txt"
    ),
    "dcrnn/sensor_graph/graph_sensor_locations.csv": (
        f"{DCRNN_GRAPH_BASE}/graph_sensor_locations.csv"
    ),
    "dcrnn/sensor_graph/graph_sensor_locations_bay.csv": (
        f"{DCRNN_GRAPH_BASE}/graph_sensor_locations_bay.csv"
    ),
    "astgnn/PEMS04/PEMS04.npz": (
        "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/"
        "main/data/PEMS04/PEMS04.npz"
    ),
    "astgnn/PEMS04/PEMS04.csv": (
        "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/"
        "main/data/PEMS04/PEMS04.csv"
    ),
    "astgnn/PEMS08/PEMS08.npz": (
        "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/"
        "main/data/PEMS08/PEMS08.npz"
    ),
    "astgnn/PEMS08/PEMS08.csv": (
        "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/"
        "main/data/PEMS08/PEMS08.csv"
    ),
}

DATASETS = ("metr_la", "pems_bay", "pems04", "pems08")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_http(url: str, output: Path, force: bool) -> None:
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"Reusing {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FedSpecShield/1.0"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def download_drive(file_id: str, output: Path, force: bool) -> None:
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"Reusing {output}")
        return
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown before downloading DCRNN data") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=file_id, output=str(output), quiet=False)
    if not result or not output.exists():
        raise RuntimeError(f"Google Drive download failed for {output.name}")


def download_sources(raw_root: Path, datasets: list[str], force: bool) -> None:
    if any(name in datasets for name in DCRNN_DRIVE_FILES):
        for name in datasets:
            if name in DCRNN_DRIVE_FILES:
                file_id, filename = DCRNN_DRIVE_FILES[name]
                download_drive(file_id, raw_root / "dcrnn" / filename, force)
        for relative, url in HTTP_FILES.items():
            if relative.startswith("dcrnn/"):
                download_http(url, raw_root / relative, force)

    for name in datasets:
        marker = "PEMS04" if name == "pems04" else "PEMS08" if name == "pems08" else None
        if marker:
            for relative, url in HTTP_FILES.items():
                if marker in relative:
                    download_http(url, raw_root / relative, force)


def read_official_adjacency(path: Path) -> tuple[list[str], np.ndarray]:
    # The pickle files are loaded only from the DCRNN author's public repository.
    with path.open("rb") as stream:
        sensor_ids, _, adjacency = pickle.load(stream, encoding="latin1")
    return [str(sensor_id) for sensor_id in sensor_ids], np.asarray(adjacency)


def read_coordinates(path: Path, dataset: str) -> dict[str, tuple[float, float]]:
    if dataset == "metr_la":
        frame = pd.read_csv(path, dtype={"sensor_id": str})
        id_column, latitude, longitude = "sensor_id", "latitude", "longitude"
    else:
        frame = pd.read_csv(
            path,
            header=None,
            names=["sensor_id", "latitude", "longitude"],
            dtype={"sensor_id": str},
        )
        id_column, latitude, longitude = "sensor_id", "latitude", "longitude"
    return {
        str(row[id_column]): (float(row[latitude]), float(row[longitude]))
        for _, row in frame.iterrows()
    }


def preprocess_dcrnn(dataset: str, raw_root: Path, output_root: Path) -> dict[str, Any]:
    filename = "metr-la.h5" if dataset == "metr_la" else "pems-bay.h5"
    graph_name = "adj_mx.pkl" if dataset == "metr_la" else "adj_mx_bay.pkl"
    location_name = (
        "graph_sensor_locations.csv"
        if dataset == "metr_la"
        else "graph_sensor_locations_bay.csv"
    )
    frame = pd.read_hdf(raw_root / "dcrnn" / filename)
    column_ids = [str(column) for column in frame.columns]
    graph_ids, adjacency = read_official_adjacency(
        raw_root / "dcrnn" / "sensor_graph" / graph_name
    )
    graph_index = {sensor_id: index for index, sensor_id in enumerate(graph_ids)}
    missing_graph_ids = [sensor_id for sensor_id in column_ids if sensor_id not in graph_index]
    if missing_graph_ids:
        raise ValueError(f"Sensors missing from official adjacency: {missing_graph_ids[:5]}")
    order = np.asarray([graph_index[sensor_id] for sensor_id in column_ids])
    adjacency = adjacency[np.ix_(order, order)].astype(np.float32)

    coordinate_map = read_coordinates(
        raw_root / "dcrnn" / "sensor_graph" / location_name,
        dataset,
    )
    missing_coordinates = [sensor_id for sensor_id in column_ids if sensor_id not in coordinate_map]
    if missing_coordinates:
        raise ValueError(f"Sensors missing coordinates: {missing_coordinates[:5]}")
    coordinates = np.asarray([coordinate_map[sensor_id] for sensor_id in column_ids], dtype=np.float32)
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset} contains non-finite values")

    output = output_root / dataset
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "values.npy", values)
    np.save(output / "adjacency.npy", adjacency)
    np.save(output / "coordinates.npy", coordinates)
    return {
        "values_shape": list(values.shape),
        "adjacency_shape": list(adjacency.shape),
        "coordinates_shape": list(coordinates.shape),
    }


def edge_adjacency(path: Path, num_nodes: int) -> np.ndarray:
    edges = pd.read_csv(path)
    required = {"from", "to"}
    if not required.issubset(edges.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    sources = edges["from"].to_numpy(dtype=np.int64)
    targets = edges["to"].to_numpy(dtype=np.int64)
    if sources.min() < 0 or targets.min() < 0 or sources.max() >= num_nodes or targets.max() >= num_nodes:
        raise ValueError(f"Edge IDs in {path} do not match {num_nodes} nodes")
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency[sources, targets] = 1.0
    adjacency[targets, sources] = 1.0
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def preprocess_pems(dataset: str, raw_root: Path, output_root: Path) -> dict[str, Any]:
    source_name = dataset.upper()
    source = raw_root / "astgnn" / source_name
    with np.load(source / f"{source_name}.npz") as archive:
        if "data" not in archive:
            raise ValueError(f"Missing 'data' array in {source_name}.npz")
        values = archive["data"].astype(np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Unexpected {dataset} values shape or non-finite values")
    adjacency = edge_adjacency(source / f"{source_name}.csv", values.shape[1])
    output = output_root / dataset
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "values.npy", values)
    np.save(output / "adjacency.npy", adjacency)
    return {
        "values_shape": list(values.shape),
        "adjacency_shape": list(adjacency.shape),
        "coordinates_shape": None,
    }


def build_manifest(raw_root: Path, output_root: Path, datasets: list[str], details: dict[str, Any]) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix != ".part":
            relative = path.relative_to(raw_root).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source": HTTP_FILES.get(relative),
            }
    for dataset, (file_id, filename) in DCRNN_DRIVE_FILES.items():
        relative = f"dcrnn/{filename}"
        if relative in files:
            files[relative]["source"] = (
                f"https://drive.google.com/uc?id={file_id}"
            )
    manifest = {
        "datasets": datasets,
        "processed": details,
        "raw_files": files,
        "notes": {
            "pems04_pems08_adjacency": (
                "Symmetric binary physical-road adjacency with self loops, built from the "
                "ASTGNN from/to edge CSV. The original cost column remains in data/raw."
            ),
            "metr_la_pems_bay_adjacency": (
                "Official DCRNN adjacency, reordered to the HDF5 sensor-column order."
            ),
        },
    }
    manifest_path = output_root / "DOWNLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare public traffic benchmark data for FedSpecShield"
    )
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="Download missing source files")
    parser.add_argument("--force", action="store_true", help="Redownload existing source files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    if args.download:
        download_sources(args.raw_root, datasets, args.force)
    details: dict[str, Any] = {}
    for dataset in datasets:
        if dataset in DCRNN_DRIVE_FILES:
            details[dataset] = preprocess_dcrnn(dataset, args.raw_root, args.output_root)
        else:
            details[dataset] = preprocess_pems(dataset, args.raw_root, args.output_root)
        print(f"Prepared {dataset}: {details[dataset]}")
    build_manifest(args.raw_root, args.output_root, datasets, details)


if __name__ == "__main__":
    main()
