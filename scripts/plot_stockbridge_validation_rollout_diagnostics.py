import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data


DATASET_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_swegnn_full_v01"
)
SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"
DEFAULT_CHECKPOINT = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14b3_autoregressive_finetune_v01"
    / "milestone14b3_best_checkpoint.pt"
)
DEFAULT_EVENT_METRICS = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14b3_autoregressive_finetune_v01"
    / "milestone14b3_best_autoregressive_event_metrics.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14b4_validation_visual_diagnostic_v01"
)

NY = 305
NX = 326
NT = 25
NUM_NODES = 99430
NUM_EDGES = 396458
EXPECTED_SPLIT_COUNTS = {"train": 70, "valid": 15, "test": 15}

INPUT_FEATURES = [
    "slope_x",
    "slope_y",
    "rainfall_t",
    "rainfall_t_plus_1",
    "DEM",
    "H_t",
    "Qmag_t",
]
EDGE_FEATURES = ["dx", "dy", "edge_slope"]
OUTPUT_VARIABLES = ["H", "Qmag"]
MODES = ("teacher_forced", "autoregressive")
TIME_STEP_MINUTES = 5
CSI_THRESHOLDS = {
    "CSI_0p05": 0.05,
    "CSI_0p10": 0.10,
    "CSI_0p20": 0.20,
}

EXPECTED_STATIC_SHAPES = {
    "DEM": (NUM_NODES,),
    "slope_x": (NUM_NODES,),
    "slope_y": (NUM_NODES,),
    "pos": (NUM_NODES, 2),
    "edge_index": (2, NUM_EDGES),
    "edge_attr": (NUM_EDGES, 3),
}
EXPECTED_DYNAMIC_SHAPES = {
    "H": (NUM_NODES, NT),
    "Qmag": (NUM_NODES, NT),
    "rainfall_global": (NT,),
}
INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

TIMESTEP_METRICS_FIELDNAMES = [
    "event_id",
    "mode",
    "input_time_index",
    "target_time_index",
    "target_minutes",
    "true_H_max",
    "pred_H_max",
    "H_peak_ratio",
    "H_RMSE",
    "H_MAE",
    "Qmag_RMSE",
    "Qmag_MAE",
    "combined_MSE",
    *CSI_THRESHOLDS,
    "raw_pred_H_min",
    "raw_pred_Qmag_min",
]

EVENT_SUMMARY_FIELDNAMES = [
    "event_id",
    "mode",
    "num_steps",
    "max_true_H_over_eval",
    "max_pred_H_over_eval",
    "H_peak_ratio",
    "mean_H_RMSE",
    "mean_H_MAE",
    "mean_Qmag_RMSE",
    "mean_Qmag_MAE",
    "mean_combined_MSE",
    "final_H_RMSE",
    "final_H_MAE",
    "final_Qmag_RMSE",
    "final_Qmag_MAE",
    "final_combined_MSE",
    "mean_CSI_0p05",
    "mean_CSI_0p10",
    "mean_CSI_0p20",
    "min_raw_pred_H",
    "min_raw_pred_Qmag",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for milestone 14B-4."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected Stockbridge validation events and create "
            "Milestone 14B-4 rollout visual diagnostics."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--event-metrics-csv",
        type=Path,
        default=DEFAULT_EVENT_METRICS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow full-graph diagnostic evaluation on CPU.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-time-index", type=int, default=2)
    parser.add_argument("--end-time-index", type=int, default=16)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--event-ids", default="auto")
    parser.add_argument(
        "--modes",
        default="teacher_forced,autoregressive",
    )
    parser.add_argument(
        "--clamp-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clamp-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-depth-vmax", type=float, default=0.0)
    parser.add_argument("--error-vmax", type=float, default=0.0)
    return parser.parse_args()


def select_device(requested_device: str, allow_cpu: bool) -> torch.device:
    """Resolve the device and protect against accidental CPU evaluation."""
    if requested_device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = requested_device

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    if device_type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Refusing to run full-graph milestone 14B-4 "
            "diagnostics on CPU unless --allow-cpu is set."
        )
    if device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Assert that a tensor contains neither NaN nor infinite values."""
    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN/inf values"


def require_keys(mapping: dict, required_keys: tuple[str, ...], label: str) -> None:
    """Raise a clear error when a serialized dictionary lacks required keys."""
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise KeyError(f"{label} is missing required keys: {missing}")


def validate_metadata_value(
    mapping: dict,
    key: str,
    expected_value,
    label: str,
) -> None:
    """Validate one scalar or list metadata value."""
    actual_value = mapping[key]
    if actual_value != expected_value:
        raise ValueError(
            f"Expected {label} {key}={expected_value!r}, "
            f"got {actual_value!r}"
        )


def load_static_graph(dataset_root: Path) -> dict:
    """Load and validate the shared Milestone 13A static graph."""
    static_graph_path = dataset_root / "static_graph.pt"
    if not static_graph_path.is_file():
        raise FileNotFoundError(f"Static graph not found: {static_graph_path}")
    static_graph = torch.load(static_graph_path, map_location="cpu")
    if not isinstance(static_graph, dict):
        raise TypeError(
            "Expected static_graph.pt to contain a dictionary, got "
            f"{type(static_graph).__name__}"
        )

    required_keys = (
        "ny",
        "nx",
        "nt",
        "num_nodes",
        "num_edges",
        "cell_size",
        "DEM",
        "slope_x",
        "slope_y",
        "pos",
        "edge_index",
        "edge_attr",
        "input_features",
        "edge_features",
        "output_variables",
    )
    require_keys(static_graph, required_keys, "static graph")
    for key, expected_value in {
        "ny": NY,
        "nx": NX,
        "nt": NT,
        "num_nodes": NUM_NODES,
        "num_edges": NUM_EDGES,
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
    }.items():
        validate_metadata_value(static_graph, key, expected_value, "static graph")

    cell_size = static_graph["cell_size"]
    if not isinstance(cell_size, (int, float)):
        raise TypeError(
            f"Expected numeric static graph cell_size, got "
            f"{type(cell_size).__name__}"
        )
    if not math.isfinite(float(cell_size)) or float(cell_size) <= 0.0:
        raise ValueError(
            f"Expected finite positive static graph cell_size, got {cell_size}"
        )
    for name, expected_shape in EXPECTED_STATIC_SHAPES.items():
        tensor = static_graph[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected static graph {name} tensor, "
                f"got {type(tensor).__name__}"
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected static graph {name} shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
    for name in ("DEM", "slope_x", "slope_y", "pos", "edge_attr"):
        assert_finite(f"static graph {name}", static_graph[name])
    edge_index = static_graph["edge_index"]
    if edge_index.dtype not in INTEGER_DTYPES:
        raise TypeError(f"Expected integer edge_index, got {edge_index.dtype}")
    row, col = edge_index
    if (row == col).any().item():
        raise ValueError("Static graph edge_index contains self-loops")
    return static_graph


def load_manifest(dataset_root: Path) -> dict:
    """Load and validate both Milestone 13A manifest files."""
    manifest_json_path = dataset_root / "manifest.json"
    manifest_csv_path = dataset_root / "manifest.csv"
    if not manifest_json_path.is_file():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_json_path}")
    if not manifest_csv_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv_path}")

    with manifest_json_path.open() as manifest_json_file:
        manifest = json.load(manifest_json_file)
    if not isinstance(manifest, dict):
        raise TypeError(
            f"Expected manifest.json object, got {type(manifest).__name__}"
        )
    required_keys = (
        "total_events",
        "split_counts",
        "contains_R019_in_test",
        "num_nodes",
        "num_edges",
        "input_features",
        "output_variables",
    )
    require_keys(manifest, required_keys, "manifest JSON")
    for key, expected_value in {
        "total_events": sum(EXPECTED_SPLIT_COUNTS.values()),
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "contains_R019_in_test": True,
        "num_nodes": NUM_NODES,
        "num_edges": NUM_EDGES,
        "input_features": INPUT_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
    }.items():
        validate_metadata_value(manifest, key, expected_value, "manifest JSON")
    if "edge_features" in manifest:
        validate_metadata_value(
            manifest,
            "edge_features",
            EDGE_FEATURES,
            "manifest JSON",
        )

    with manifest_csv_path.open(newline="") as manifest_csv_file:
        manifest_rows = list(csv.DictReader(manifest_csv_file))
    expected_total_events = sum(EXPECTED_SPLIT_COUNTS.values())
    if len(manifest_rows) != expected_total_events:
        raise ValueError(
            f"Expected {expected_total_events} rows in manifest.csv, "
            f"found {len(manifest_rows)}"
        )
    return manifest


def load_dynamic_event(
    dataset_root: Path,
    split: str,
    event_id: str,
) -> dict:
    """Load and validate one Milestone 13A dynamic event dictionary."""
    event_path = dataset_root / "events" / split / f"{event_id}.pt"
    if not event_path.is_file():
        raise FileNotFoundError(
            f"Dynamic event file not found for {split}/{event_id}: {event_path}"
        )
    dynamic_event = torch.load(event_path, map_location="cpu")
    if not isinstance(dynamic_event, dict):
        raise TypeError(
            f"Expected {event_path} dictionary, got "
            f"{type(dynamic_event).__name__}"
        )
    required_keys = (
        "event_id",
        "split",
        "ny",
        "nx",
        "nt",
        "num_nodes",
        "H",
        "Qmag",
        "rainfall_global",
    )
    require_keys(dynamic_event, required_keys, f"dynamic event {split}/{event_id}")
    for key, expected_value in {
        "event_id": event_id,
        "split": split,
        "ny": NY,
        "nx": NX,
        "nt": NT,
        "num_nodes": NUM_NODES,
    }.items():
        validate_metadata_value(
            dynamic_event,
            key,
            expected_value,
            f"dynamic event {split}/{event_id}",
        )
    for name, expected_shape in EXPECTED_DYNAMIC_SHAPES.items():
        tensor = dynamic_event[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected {split}/{event_id} {name} tensor, "
                f"got {type(tensor).__name__}"
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected {split}/{event_id} {name} shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        assert_finite(f"{split}/{event_id} {name}", tensor)
    H_min = dynamic_event["H"].min().item()
    Qmag_min = dynamic_event["Qmag"].min().item()
    if H_min < -1e-8:
        raise ValueError(f"{split}/{event_id} H minimum {H_min:.8g} is below -1e-8")
    if Qmag_min < -1e-8:
        raise ValueError(
            f"{split}/{event_id} Qmag minimum {Qmag_min:.8g} is below -1e-8"
        )
    return dynamic_event


def static_graph_to_device(static_graph: dict, device: torch.device) -> dict:
    """Move the shared static graph tensors to the device exactly once."""
    device_graph = static_graph.copy()
    for name in ("DEM", "slope_x", "slope_y", "pos", "edge_attr"):
        device_graph[name] = static_graph[name].float().to(device)
    device_graph["edge_index"] = static_graph["edge_index"].long().to(device)
    return device_graph


def natural_event_sort_key(event_id: str) -> tuple[int, str]:
    """Sort canonical Stockbridge IDs numerically, with a stable fallback."""
    match = re.fullmatch(r"R(\d+)", event_id)
    if match is None:
        raise ValueError(f"Invalid Stockbridge event ID: {event_id!r}")
    return int(match.group(1)), event_id


def derive_split_event_ids(
    dataset_root: Path,
    manifest: dict,
    split: str,
) -> list[str]:
    """Derive naturally sorted event IDs from the manifest or event files."""
    event_ids_by_split = manifest.get("event_ids_by_split")
    if event_ids_by_split is not None:
        if not isinstance(event_ids_by_split, dict):
            raise TypeError("manifest event_ids_by_split must be a dictionary")
        if split not in event_ids_by_split:
            raise KeyError(f"manifest event_ids_by_split is missing {split!r}")
        event_ids = event_ids_by_split[split]
        if not isinstance(event_ids, list) or not all(
            isinstance(event_id, str) for event_id in event_ids
        ):
            raise TypeError(
                f"manifest event_ids_by_split[{split!r}] must be a string list"
            )
    else:
        split_dir = dataset_root / "events" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Event split directory not found: {split_dir}")
        event_ids = [path.stem for path in split_dir.glob("*.pt")]

    sorted_event_ids = sorted(event_ids, key=natural_event_sort_key)
    if len(set(sorted_event_ids)) != len(sorted_event_ids):
        raise ValueError(f"Duplicate event IDs in {split} split")
    expected_count = EXPECTED_SPLIT_COUNTS[split]
    if len(sorted_event_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {split} event IDs, "
            f"found {len(sorted_event_ids)}"
        )
    missing_paths = [
        dataset_root / "events" / split / f"{event_id}.pt"
        for event_id in sorted_event_ids
        if not (dataset_root / "events" / split / f"{event_id}.pt").is_file()
    ]
    if missing_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing {split} dynamic event files:\n{formatted}")
    return sorted_event_ids


def parse_modes(modes_text: str) -> list[str]:
    """Parse an ordered, duplicate-free list of supported evaluation modes."""
    modes = [value.strip() for value in modes_text.split(",") if value.strip()]
    if not modes:
        raise ValueError("--modes must contain at least one evaluation mode")
    unsupported = [mode for mode in modes if mode not in MODES]
    if unsupported:
        raise ValueError(
            f"Unsupported --modes values {unsupported}; allowed modes are {MODES}"
        )
    if len(set(modes)) != len(modes):
        raise ValueError(f"--modes contains duplicates: {modes}")
    return modes


def parse_event_ids(event_ids_text: str) -> list[str]:
    """Parse explicit event IDs while preserving the requested order."""
    event_ids = [
        value.strip()
        for value in event_ids_text.split(",")
        if value.strip()
    ]
    if not event_ids:
        raise ValueError("--event-ids must be 'auto' or a nonempty ID list")
    for event_id in event_ids:
        natural_event_sort_key(event_id)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError(f"--event-ids contains duplicates: {event_ids}")
    return event_ids


def parse_metric_float(
    row: dict[str, str],
    field: str,
    event_id: str,
    allow_blank: bool = False,
) -> float | None:
    """Parse one finite numeric event-metric field with row context."""
    raw_value = row.get(field, "").strip()
    if not raw_value:
        if allow_blank:
            return None
        raise ValueError(
            f"Missing {field} for {event_id} in event metrics CSV"
        )
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Invalid {field}={raw_value!r} for {event_id}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field}={value} for {event_id}")
    return value


def select_representative_events(
    event_metrics_csv: Path,
    valid_event_ids: list[str],
) -> list[str]:
    """Select best, median, worst, under-, and overpredicted events."""
    if not event_metrics_csv.is_file():
        raise FileNotFoundError(
            f"Event metrics CSV not found: {event_metrics_csv}"
        )
    with event_metrics_csv.open(newline="") as metrics_file:
        reader = csv.DictReader(metrics_file)
        required_fields = {"event_id", "mean_combined_MSE", "H_peak_ratio"}
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                "Event metrics CSV is missing fields: "
                f"{sorted(missing_fields)}"
            )
        rows = [
            row
            for row in reader
            if "mode" not in row
            or row["mode"].strip() == "autoregressive"
        ]
    if not rows:
        raise ValueError("Event metrics CSV has no autoregressive event rows")

    valid_set = set(valid_event_ids)
    records = []
    seen_event_ids = set()
    for row in rows:
        event_id = row["event_id"].strip()
        natural_event_sort_key(event_id)
        if event_id not in valid_set:
            raise ValueError(
                f"Event metrics CSV contains non-validation event {event_id}"
            )
        if event_id in seen_event_ids:
            raise ValueError(
                f"Event metrics CSV contains duplicate event {event_id}"
            )
        seen_event_ids.add(event_id)
        mean_combined_MSE = parse_metric_float(
            row,
            "mean_combined_MSE",
            event_id,
        )
        if mean_combined_MSE < 0.0:
            raise ValueError(
                f"mean_combined_MSE must be nonnegative for {event_id}"
            )
        records.append(
            {
                "event_id": event_id,
                "mean_combined_MSE": mean_combined_MSE,
                "H_peak_ratio": parse_metric_float(
                    row,
                    "H_peak_ratio",
                    event_id,
                    allow_blank=True,
                ),
            }
        )
    if seen_event_ids != valid_set:
        missing_event_ids = sorted(
            valid_set.difference(seen_event_ids),
            key=natural_event_sort_key,
        )
        raise ValueError(
            "Event metrics CSV does not cover all validation events; missing "
            f"{missing_event_ids}"
        )

    by_loss = sorted(
        records,
        key=lambda row: (
            row["mean_combined_MSE"],
            natural_event_sort_key(row["event_id"]),
        ),
    )
    with_peak_ratio = [
        row for row in records if row["H_peak_ratio"] is not None
    ]
    if not with_peak_ratio:
        raise ValueError(
            "Event metrics CSV has no defined H_peak_ratio values"
        )
    by_peak_ratio = sorted(
        with_peak_ratio,
        key=lambda row: (
            row["H_peak_ratio"],
            natural_event_sort_key(row["event_id"]),
        ),
    )
    candidates = [
        by_loss[0]["event_id"],
        by_loss[len(by_loss) // 2]["event_id"],
        by_loss[-1]["event_id"],
        by_peak_ratio[0]["event_id"],
        by_peak_ratio[-1]["event_id"],
    ]
    return list(dict.fromkeys(candidates))


def resolve_selected_event_ids(
    event_ids_text: str,
    event_metrics_csv: Path,
    valid_event_ids: list[str],
) -> tuple[list[str], str]:
    """Resolve automatic or explicit validation-event selection."""
    if event_ids_text.strip().lower() == "auto":
        return (
            select_representative_events(event_metrics_csv, valid_event_ids),
            "auto_event_metrics_representatives",
        )
    selected_event_ids = parse_event_ids(event_ids_text)
    invalid = [
        event_id
        for event_id in selected_event_ids
        if event_id not in set(valid_event_ids)
    ]
    if invalid:
        raise ValueError(
            f"Selected event IDs are not in the validation split: {invalid}"
        )
    if "R019" in selected_event_ids:
        raise ValueError("R019 must not be selected for validation diagnostics")
    return selected_event_ids, "explicit_event_ids"


def prepare_dynamic_events(
    dataset_root: Path,
    split: str,
    event_ids: list[str],
    device: torch.device,
) -> list[dict]:
    """Load each selected dynamic event and move its tensors once."""
    prepared_events = []
    print(f"Preparing {split} events on {device}: 0/{len(event_ids)}")
    for event_number, event_id in enumerate(event_ids, start=1):
        dynamic_event = load_dynamic_event(dataset_root, split, event_id)
        prepared_event = {
            "event_id": event_id,
            "split": split,
            "H": dynamic_event["H"].float().to(device),
            "Qmag": dynamic_event["Qmag"].float().to(device),
            "rainfall_global": dynamic_event["rainfall_global"].float().to(device),
        }
        for name in ("H", "Qmag", "rainfall_global"):
            assert prepared_event[name].device == device
        prepared_events.append(prepared_event)
        if event_number % 10 == 0 or event_number == len(event_ids):
            print(
                f"Preparing {split} events on {device}: "
                f"{event_number}/{len(event_ids)}"
            )
    return prepared_events


def build_device_sample_from_prepared(
    static_device_graph: dict,
    prepared_event: dict,
    t: int,
    rainfall_scale: float,
) -> Data:
    """Build a one-step sample without repeating device transfers."""
    if t < 0 or t >= NT - 1:
        raise ValueError(f"time index must satisfy 0 <= t < {NT - 1}; got {t}")
    if not math.isfinite(rainfall_scale):
        raise ValueError("rainfall_scale must be finite")
    device = static_device_graph["DEM"].device
    for name in ("H", "Qmag", "rainfall_global"):
        if prepared_event[name].device != device:
            raise ValueError(
                f"Prepared {prepared_event['event_id']} {name} is not on {device}"
            )
    rainfall_global = prepared_event["rainfall_global"]
    rainfall_t_scaled = rainfall_global[t] * rainfall_scale
    rainfall_next_scaled = rainfall_global[t + 1] * rainfall_scale
    H_t = prepared_event["H"][:, t]
    Qmag_t = prepared_event["Qmag"][:, t]
    x = torch.stack(
        [
            static_device_graph["slope_x"],
            static_device_graph["slope_y"],
            rainfall_t_scaled.expand(NUM_NODES),
            rainfall_next_scaled.expand(NUM_NODES),
            static_device_graph["DEM"],
            H_t,
            Qmag_t,
        ],
        dim=1,
    )
    y = torch.stack(
        [
            prepared_event["H"][:, t + 1],
            prepared_event["Qmag"][:, t + 1],
        ],
        dim=1,
    )

    sample = Data()
    sample.x = x
    sample.y = y
    sample.edge_index = static_device_graph["edge_index"]
    sample.edge_attr = static_device_graph["edge_attr"]
    sample.pos = static_device_graph["pos"]
    sample.DEM = static_device_graph["DEM"]
    sample.event_id = prepared_event["event_id"]
    sample.split = prepared_event["split"]
    sample.input_time_index = t
    sample.target_time_index = t + 1
    sample.rainfall_input_scaled = rainfall_t_scaled
    sample.rainfall_target_scaled = rainfall_next_scaled
    sample.input_features = INPUT_FEATURES
    sample.edge_features = EDGE_FEATURES
    sample.output_variables = OUTPUT_VARIABLES
    sample.previous_t = 1
    sample.ny = NY
    sample.nx = NX
    sample.nt = NT
    sample.num_nodes = NUM_NODES

    assert tuple(sample.x.shape) == (NUM_NODES, 7)
    assert tuple(sample.y.shape) == (NUM_NODES, 2)
    assert tuple(sample.edge_index.shape) == (2, NUM_EDGES)
    assert tuple(sample.edge_attr.shape) == (NUM_EDGES, 3)
    assert sample.edge_index.data_ptr() == static_device_graph[
        "edge_index"
    ].data_ptr()
    assert sample.edge_attr.data_ptr() == static_device_graph[
        "edge_attr"
    ].data_ptr()
    assert torch.equal(sample.x[:, -3], static_device_graph["DEM"])
    assert torch.equal(sample.x[:, -2], H_t)
    assert torch.equal(sample.x[:, -1], Qmag_t)
    assert INPUT_FEATURES[-3:] == ["DEM", "H_t", "Qmag_t"]
    assert_finite(f"{prepared_event['event_id']} sample x", sample.x)
    assert_finite(f"{prepared_event['event_id']} sample y", sample.y)
    return sample


def extract_prediction(model_output, context: str) -> torch.Tensor:
    """Extract a prediction tensor from direct or tuple/list model output."""
    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise ValueError(f"Model returned an empty tuple/list during {context}")
        model_output = model_output[0]
    if not isinstance(model_output, torch.Tensor):
        raise TypeError(
            f"Expected prediction tensor during {context}, "
            f"got {type(model_output).__name__}"
        )
    return model_output


def safe_peak_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite peak ratio, or None when the true peak is zero."""
    if denominator == 0.0:
        return None
    ratio = numerator / denominator
    if not math.isfinite(ratio):
        raise ValueError("Non-finite predicted/true H peak ratio")
    return ratio


def mean_optional(values: list[float | None]) -> float | None:
    """Average defined values, retaining None when all are undefined."""
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def mean_required(values: list[float], label: str) -> float:
    """Average a non-empty list and validate the result."""
    if not values:
        raise ValueError(f"Cannot average empty values for {label}")
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite mean for {label}: {result}")
    return result


def scalar_metadata(value, label: str):
    """Convert scalar checkpoint metadata to a Python scalar."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Expected scalar checkpoint {label}, got shape "
                f"{tuple(value.shape)}"
            )
        value = value.item()
    return value


def load_and_validate_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, float | int | str]]:
    """Load and validate a Milestone 14B-3 or fallback 14B-1 checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dictionary, got "
            f"{type(checkpoint).__name__}"
        )
    required_keys = (
        "model_state_dict",
        "input_features",
        "edge_features",
        "output_variables",
        "model_config",
        "rainfall_scale",
        "start_time_index",
        "end_time_index",
        "best_epoch",
    )
    require_keys(checkpoint, required_keys, "checkpoint")
    if "best_valid_autoreg_loss" in checkpoint:
        checkpoint_type = "14B-3"
        best_loss_key = "best_valid_autoreg_loss"
    elif "best_valid_loss" in checkpoint:
        checkpoint_type = "14B-1"
        best_loss_key = "best_valid_loss"
    else:
        raise KeyError(
            "checkpoint must contain best_valid_autoreg_loss (14B-3) or "
            "best_valid_loss (14B-1)"
        )
    validate_metadata_value(
        checkpoint,
        "input_features",
        INPUT_FEATURES,
        "checkpoint",
    )
    validate_metadata_value(
        checkpoint,
        "edge_features",
        EDGE_FEATURES,
        "checkpoint",
    )
    validate_metadata_value(
        checkpoint,
        "output_variables",
        OUTPUT_VARIABLES,
        "checkpoint",
    )
    if not isinstance(checkpoint["model_state_dict"], dict):
        raise TypeError("checkpoint model_state_dict must be a dictionary")
    model_config = checkpoint["model_config"]
    if not isinstance(model_config, dict):
        raise TypeError("checkpoint model_config must be a dictionary")

    expected_model_config = {
        "node_features": 7,
        "edge_features": 3,
        "type_GNN": "SWEGNN",
        "hid_features": args.hid_features,
        "K": args.K,
        "gnn_activation": "tanh",
        "dropout": 0.0,
        "mlp_layers": 2,
        "mlp_activation": "prelu",
        "seed": args.seed,
        "with_filter_matrix": True,
        "with_gradient": True,
        "with_WL": True,
        "previous_t": 1,
    }
    mismatches = {
        name: (model_config.get(name), expected_value)
        for name, expected_value in expected_model_config.items()
        if model_config.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(
            "Checkpoint model_config does not match evaluation model: "
            f"{mismatches}"
        )

    checkpoint_rainfall_scale = float(
        scalar_metadata(checkpoint["rainfall_scale"], "rainfall_scale")
    )
    checkpoint_start_time_index = int(
        scalar_metadata(checkpoint["start_time_index"], "start_time_index")
    )
    checkpoint_end_time_index = int(
        scalar_metadata(checkpoint["end_time_index"], "end_time_index")
    )
    checkpoint_best_loss = float(
        scalar_metadata(checkpoint[best_loss_key], best_loss_key)
    )
    checkpoint_best_epoch = int(
        scalar_metadata(checkpoint["best_epoch"], "best_epoch")
    )
    if not math.isfinite(checkpoint_rainfall_scale):
        raise ValueError("checkpoint rainfall_scale must be finite")
    if checkpoint_start_time_index < 0:
        raise ValueError("checkpoint start_time_index must be nonnegative")
    if checkpoint_end_time_index >= NT - 1:
        raise ValueError(
            f"checkpoint end_time_index must be less than {NT - 1}"
        )
    if checkpoint_start_time_index > checkpoint_end_time_index:
        raise ValueError(
            "checkpoint start_time_index must not exceed end_time_index"
        )
    if (
        not math.isfinite(checkpoint_best_loss)
        or checkpoint_best_loss < 0.0
    ):
        raise ValueError(
            f"checkpoint {best_loss_key} must be finite and nonnegative"
        )
    if checkpoint_best_epoch <= 0:
        raise ValueError("checkpoint best_epoch must be greater than zero")

    metadata = {
        "checkpoint_type": checkpoint_type,
        "checkpoint_rainfall_scale": checkpoint_rainfall_scale,
        "checkpoint_start_time_index": checkpoint_start_time_index,
        "checkpoint_end_time_index": checkpoint_end_time_index,
        "checkpoint_best_loss": checkpoint_best_loss,
        "checkpoint_best_epoch": checkpoint_best_epoch,
    }
    return checkpoint, metadata


def build_rollout_sample(
    static_device_graph: dict,
    prepared_event: dict,
    current_H: torch.Tensor,
    current_Qmag: torch.Tensor,
    t: int,
    rainfall_scale: float,
) -> Data:
    """Build one rollout sample using the current autoregressive state."""
    if t < 0 or t >= NT - 1:
        raise ValueError(f"time index must satisfy 0 <= t < {NT - 1}; got {t}")
    if not math.isfinite(rainfall_scale):
        raise ValueError("rainfall_scale must be finite")
    if not isinstance(current_H, torch.Tensor):
        raise TypeError("current_H must be a tensor")
    if not isinstance(current_Qmag, torch.Tensor):
        raise TypeError("current_Qmag must be a tensor")
    if tuple(current_H.shape) != (NUM_NODES,):
        raise ValueError(
            f"Expected current_H shape {(NUM_NODES,)}, "
            f"got {tuple(current_H.shape)}"
        )
    if tuple(current_Qmag.shape) != (NUM_NODES,):
        raise ValueError(
            f"Expected current_Qmag shape {(NUM_NODES,)}, "
            f"got {tuple(current_Qmag.shape)}"
        )

    device = static_device_graph["DEM"].device
    for name, tensor in {
        "current_H": current_H,
        "current_Qmag": current_Qmag,
        "event H": prepared_event["H"],
        "event Qmag": prepared_event["Qmag"],
        "event rainfall_global": prepared_event["rainfall_global"],
    }.items():
        if tensor.device != device:
            raise ValueError(f"{name} is not on {device}")
    assert_finite("rollout current_H", current_H)
    assert_finite("rollout current_Qmag", current_Qmag)

    rainfall_global = prepared_event["rainfall_global"]
    rainfall_t_scaled = rainfall_global[t] * rainfall_scale
    rainfall_next_scaled = rainfall_global[t + 1] * rainfall_scale
    x = torch.stack(
        [
            static_device_graph["slope_x"],
            static_device_graph["slope_y"],
            rainfall_t_scaled.expand(NUM_NODES),
            rainfall_next_scaled.expand(NUM_NODES),
            static_device_graph["DEM"],
            current_H,
            current_Qmag,
        ],
        dim=1,
    )
    y = torch.stack(
        [
            prepared_event["H"][:, t + 1],
            prepared_event["Qmag"][:, t + 1],
        ],
        dim=1,
    )

    sample = Data()
    sample.x = x
    sample.y = y
    sample.edge_index = static_device_graph["edge_index"]
    sample.edge_attr = static_device_graph["edge_attr"]
    sample.pos = static_device_graph["pos"]
    sample.DEM = static_device_graph["DEM"]
    sample.event_id = prepared_event["event_id"]
    sample.split = prepared_event["split"]
    sample.input_time_index = t
    sample.target_time_index = t + 1
    sample.rainfall_input_scaled = rainfall_t_scaled
    sample.rainfall_target_scaled = rainfall_next_scaled
    sample.input_features = INPUT_FEATURES
    sample.edge_features = EDGE_FEATURES
    sample.output_variables = OUTPUT_VARIABLES
    sample.previous_t = 1
    sample.ny = NY
    sample.nx = NX
    sample.nt = NT
    sample.num_nodes = NUM_NODES

    expected_shapes = {
        "x": (NUM_NODES, 7),
        "y": (NUM_NODES, 2),
        "edge_index": (2, NUM_EDGES),
        "edge_attr": (NUM_EDGES, 3),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(getattr(sample, name).shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Expected rollout sample {name} shape {expected_shape}, "
                f"got {actual_shape}"
            )
    for name in ("x", "y", "edge_attr"):
        assert_finite(
            f"{prepared_event['event_id']} rollout sample {name}",
            getattr(sample, name),
        )
    if sample.edge_index.data_ptr() != static_device_graph[
        "edge_index"
    ].data_ptr():
        raise RuntimeError("Rollout sample did not reuse shared edge_index")
    if sample.edge_attr.data_ptr() != static_device_graph[
        "edge_attr"
    ].data_ptr():
        raise RuntimeError("Rollout sample did not reuse shared edge_attr")
    if not torch.equal(sample.x[:, -3], static_device_graph["DEM"]):
        raise RuntimeError("Feature -3 must be DEM")
    if not torch.equal(sample.x[:, -2], current_H):
        raise RuntimeError("Feature -2 must be current_H")
    if not torch.equal(sample.x[:, -1], current_Qmag):
        raise RuntimeError("Feature -1 must be current_Qmag")
    return sample


def calculate_csi(
    pred_H: torch.Tensor,
    true_H: torch.Tensor,
    threshold: float,
) -> float | None:
    """Calculate critical success index for one pair and threshold."""
    pred_wet = pred_H >= threshold
    true_wet = true_H >= threshold
    true_positive = torch.logical_and(pred_wet, true_wet).sum().item()
    false_positive = torch.logical_and(pred_wet, ~true_wet).sum().item()
    false_negative = torch.logical_and(~pred_wet, true_wet).sum().item()
    denominator = true_positive + false_positive + false_negative
    if denominator == 0:
        return None
    csi = true_positive / denominator
    if not math.isfinite(csi):
        raise ValueError(f"Non-finite CSI at threshold {threshold}")
    return csi


def calculate_pair_metrics(
    raw_pred: torch.Tensor,
    metric_pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | None]:
    """Calculate requested metrics for one prediction/target pair."""
    H_error = metric_pred[:, 0] - target[:, 0]
    Qmag_error = metric_pred[:, 1] - target[:, 1]
    H_MSE = torch.mean(H_error.square())
    Qmag_MSE = torch.mean(Qmag_error.square())
    combined_MSE = torch.mean((metric_pred - target).square())
    tensor_metrics = {
        "H_MSE": H_MSE,
        "Qmag_MSE": Qmag_MSE,
        "combined_MSE": combined_MSE,
    }
    for name, tensor in tensor_metrics.items():
        assert_finite(f"pair metric {name}", tensor)

    metrics: dict[str, float | None] = {
        "H_RMSE": torch.sqrt(H_MSE).item(),
        "H_MAE": torch.mean(torch.abs(H_error)).item(),
        "Qmag_RMSE": torch.sqrt(Qmag_MSE).item(),
        "Qmag_MAE": torch.mean(torch.abs(Qmag_error)).item(),
        "combined_MSE": combined_MSE.item(),
        "true_H_max": target[:, 0].max().item(),
        "pred_H_max": metric_pred[:, 0].max().item(),
        "raw_pred_H_min": raw_pred[:, 0].min().item(),
        "raw_pred_Qmag_min": raw_pred[:, 1].min().item(),
    }
    for label, threshold in CSI_THRESHOLDS.items():
        metrics[label] = calculate_csi(
            metric_pred[:, 0],
            target[:, 0],
            threshold,
        )
    for name, value in metrics.items():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"Non-finite pair metric {name}={value}")
    return metrics


def summarize_event_metrics(
    event_id: str,
    mode: str,
    timestep_metrics: list[dict[str, float | int | str | None]],
) -> dict[str, float | int | str | None]:
    """Aggregate the requested per-event diagnostics."""
    if not timestep_metrics:
        raise ValueError(f"No {mode} timestep metrics for {event_id}")
    final_metrics = timestep_metrics[-1]
    max_true_H = max(
        float(row["true_H_max"]) for row in timestep_metrics
    )
    max_pred_H = max(
        float(row["pred_H_max"]) for row in timestep_metrics
    )
    summary: dict[str, float | int | str | None] = {
        "event_id": event_id,
        "mode": mode,
        "num_steps": len(timestep_metrics),
        "max_true_H_over_eval": max_true_H,
        "max_pred_H_over_eval": max_pred_H,
        "H_peak_ratio": safe_peak_ratio(max_pred_H, max_true_H),
        "mean_H_RMSE": mean_required(
            [float(row["H_RMSE"]) for row in timestep_metrics],
            f"{event_id} {mode} H_RMSE",
        ),
        "mean_H_MAE": mean_required(
            [float(row["H_MAE"]) for row in timestep_metrics],
            f"{event_id} {mode} H_MAE",
        ),
        "mean_Qmag_RMSE": mean_required(
            [float(row["Qmag_RMSE"]) for row in timestep_metrics],
            f"{event_id} {mode} Qmag_RMSE",
        ),
        "mean_Qmag_MAE": mean_required(
            [float(row["Qmag_MAE"]) for row in timestep_metrics],
            f"{event_id} {mode} Qmag_MAE",
        ),
        "mean_combined_MSE": mean_required(
            [float(row["combined_MSE"]) for row in timestep_metrics],
            f"{event_id} {mode} combined_MSE",
        ),
        "final_H_RMSE": float(final_metrics["H_RMSE"]),
        "final_H_MAE": float(final_metrics["H_MAE"]),
        "final_Qmag_RMSE": float(final_metrics["Qmag_RMSE"]),
        "final_Qmag_MAE": float(final_metrics["Qmag_MAE"]),
        "final_combined_MSE": float(final_metrics["combined_MSE"]),
        "mean_CSI_0p05": mean_optional(
            [row["CSI_0p05"] for row in timestep_metrics]
        ),
        "mean_CSI_0p10": mean_optional(
            [row["CSI_0p10"] for row in timestep_metrics]
        ),
        "mean_CSI_0p20": mean_optional(
            [row["CSI_0p20"] for row in timestep_metrics]
        ),
        "min_raw_pred_H": min(
            float(row["raw_pred_H_min"]) for row in timestep_metrics
        ),
        "min_raw_pred_Qmag": min(
            float(row["raw_pred_Qmag_min"]) for row in timestep_metrics
        ),
    }
    for name, value in summary.items():
        if name not in ("event_id", "mode") and value is not None:
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Non-finite event summary {event_id} {mode} "
                    f"{name}={value}"
                )
    return summary


def build_prediction_bundle(
    event: dict,
    mode: str,
    time_indices: list[int],
    target_time_indices: list[int],
    target_minutes: list[int],
    pred_H_steps: list[torch.Tensor],
    pred_Qmag_steps: list[torch.Tensor],
    true_H_steps: list[torch.Tensor],
    true_Qmag_steps: list[torch.Tensor],
    rainfall_scale: float,
    clamp_output: bool,
    clamp_feedback: bool,
) -> dict:
    """Build and validate one CPU prediction bundle."""
    prediction_data = {
        "event_id": event["event_id"],
        "mode": mode,
        "start_time_index": time_indices[0],
        "end_time_index": time_indices[-1],
        "target_time_indices": target_time_indices,
        "target_minutes": target_minutes,
        "pred_H": torch.stack(pred_H_steps, dim=0),
        "pred_Qmag": torch.stack(pred_Qmag_steps, dim=0),
        "true_H": torch.stack(true_H_steps, dim=0),
        "true_Qmag": torch.stack(true_Qmag_steps, dim=0),
        "rainfall_global": event["rainfall_global"].detach().cpu(),
        "rainfall_scale": rainfall_scale,
        "clamp_output": clamp_output,
        "clamp_feedback": clamp_feedback,
        "input_features": INPUT_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
    }
    expected_shape = (len(time_indices), NUM_NODES)
    for name in ("pred_H", "pred_Qmag", "true_H", "true_Qmag"):
        tensor = prediction_data[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected {event['event_id']} {mode} {name} shape "
                f"{expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.device.type != "cpu":
            raise ValueError(f"{event['event_id']} {mode} {name} is not on CPU")
        assert_finite(f"{event['event_id']} {mode} {name}", tensor)
    rainfall_global = prediction_data["rainfall_global"]
    if tuple(rainfall_global.shape) != (NT,):
        raise ValueError(
            f"Expected rainfall_global shape {(NT,)}, "
            f"got {tuple(rainfall_global.shape)}"
        )
    if rainfall_global.device.type != "cpu":
        raise ValueError("Saved rainfall_global is not on CPU")
    assert_finite("saved rainfall_global", rainfall_global)
    return prediction_data


def evaluate_event_mode(
    model: torch.nn.Module,
    static_device_graph: dict,
    event: dict,
    time_indices: list[int],
    rainfall_scale: float,
    mode: str,
    clamp_output: bool,
    clamp_feedback: bool,
) -> tuple[
    list[dict[str, float | int | str | None]],
    dict[str, float | int | str | None],
    dict,
]:
    """Evaluate one selected validation event in one inference mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown evaluation mode {mode!r}")
    if not time_indices:
        raise ValueError("Evaluation time_indices must not be empty")

    target_time_indices = [time_index + 1 for time_index in time_indices]
    target_minutes = [
        time_index * TIME_STEP_MINUTES for time_index in target_time_indices
    ]
    current_H = event["H"][:, time_indices[0]]
    current_Qmag = event["Qmag"][:, time_indices[0]]
    timestep_metrics = []
    pred_H_steps = []
    pred_Qmag_steps = []
    true_H_steps = []
    true_Qmag_steps = []
    model.eval()

    with torch.inference_mode():
        for t, target_time_index, target_minute in zip(
            time_indices,
            target_time_indices,
            target_minutes,
            strict=True,
        ):
            if mode == "teacher_forced":
                sample = build_device_sample_from_prepared(
                    static_device_graph,
                    event,
                    t,
                    rainfall_scale,
                )
            else:
                sample = build_rollout_sample(
                    static_device_graph,
                    event,
                    current_H,
                    current_Qmag,
                    t,
                    rainfall_scale,
                )
            context = f"{mode} evaluation for {event['event_id']} at T{t}"
            raw_pred = extract_prediction(model(sample), context)
            if tuple(raw_pred.shape) != (NUM_NODES, 2):
                raise ValueError(
                    f"Expected prediction shape {(NUM_NODES, 2)} during "
                    f"{context}, got {tuple(raw_pred.shape)}"
                )
            assert_finite(
                f"{event['event_id']} {mode} raw prediction at T{t}",
                raw_pred,
            )
            metric_pred = raw_pred.clamp_min(0.0) if clamp_output else raw_pred
            pair_metrics = calculate_pair_metrics(
                raw_pred,
                metric_pred,
                sample.y,
            )
            timestep_metrics.append(
                {
                    "event_id": event["event_id"],
                    "mode": mode,
                    "input_time_index": t,
                    "target_time_index": target_time_index,
                    "target_minutes": target_minute,
                    "true_H_max": pair_metrics["true_H_max"],
                    "pred_H_max": pair_metrics["pred_H_max"],
                    "H_peak_ratio": safe_peak_ratio(
                        float(pair_metrics["pred_H_max"]),
                        float(pair_metrics["true_H_max"]),
                    ),
                    "H_RMSE": pair_metrics["H_RMSE"],
                    "H_MAE": pair_metrics["H_MAE"],
                    "Qmag_RMSE": pair_metrics["Qmag_RMSE"],
                    "Qmag_MAE": pair_metrics["Qmag_MAE"],
                    "combined_MSE": pair_metrics["combined_MSE"],
                    "CSI_0p05": pair_metrics["CSI_0p05"],
                    "CSI_0p10": pair_metrics["CSI_0p10"],
                    "CSI_0p20": pair_metrics["CSI_0p20"],
                    "raw_pred_H_min": pair_metrics["raw_pred_H_min"],
                    "raw_pred_Qmag_min": pair_metrics[
                        "raw_pred_Qmag_min"
                    ],
                }
            )
            pred_H_steps.append(metric_pred[:, 0].detach().cpu())
            pred_Qmag_steps.append(metric_pred[:, 1].detach().cpu())
            true_H_steps.append(sample.y[:, 0].detach().cpu())
            true_Qmag_steps.append(sample.y[:, 1].detach().cpu())

            if mode == "autoregressive":
                feedback_state = raw_pred
                if clamp_feedback:
                    feedback_state = feedback_state.clamp_min(0.0)
                current_H = feedback_state[:, 0].detach()
                current_Qmag = feedback_state[:, 1].detach()

    summary = summarize_event_metrics(
        event["event_id"],
        mode,
        timestep_metrics,
    )
    prediction_data = build_prediction_bundle(
        event,
        mode,
        time_indices,
        target_time_indices,
        target_minutes,
        pred_H_steps,
        pred_Qmag_steps,
        true_H_steps,
        true_Qmag_steps,
        rainfall_scale,
        clamp_output,
        clamp_feedback,
    )
    return timestep_metrics, summary, prediction_data


def write_metrics_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    """Write a nonempty metrics table with a fixed column order."""
    if not rows:
        raise ValueError(f"Cannot write empty metrics CSV: {output_path}")
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def automatic_vmax(tensors: tuple[torch.Tensor, ...], label: str) -> float:
    """Return a finite positive automatic map limit."""
    vmax = max(float(tensor.max().item()) for tensor in tensors)
    if not math.isfinite(vmax):
        raise ValueError(f"Non-finite automatic vmax for {label}")
    return vmax if vmax > 0.0 else 1.0


def diagnostic_figure_title(
    event_id: str,
    mode: str,
    checkpoint_path: Path,
    start_time_index: int,
    end_time_index: int,
) -> str:
    """Build the shared map title required for diagnostic context."""
    return (
        f"{event_id} | {mode} | {checkpoint_path.name} | "
        f"T{start_time_index}->T{start_time_index + 1} through "
        f"T{end_time_index}->T{end_time_index + 1}"
    )


def plot_depth_comparison(
    true_H: torch.Tensor,
    pred_H: torch.Tensor,
    output_path: Path,
    panel_prefix: str,
    figure_title: str,
    requested_depth_vmax: float,
    requested_error_vmax: float,
    dpi: int,
) -> None:
    """Plot true depth, predicted depth, and their absolute error."""
    if tuple(true_H.shape) != (NUM_NODES,):
        raise ValueError(f"Expected true depth shape {(NUM_NODES,)}")
    if tuple(pred_H.shape) != (NUM_NODES,):
        raise ValueError(f"Expected predicted depth shape {(NUM_NODES,)}")
    assert_finite("plotted true depth", true_H)
    assert_finite("plotted predicted depth", pred_H)
    absolute_error = torch.abs(pred_H - true_H)
    depth_vmax = (
        requested_depth_vmax
        if requested_depth_vmax > 0.0
        else automatic_vmax((true_H, pred_H), "depth")
    )
    error_vmax = (
        requested_error_vmax
        if requested_error_vmax > 0.0
        else automatic_vmax((absolute_error,), "absolute depth error")
    )

    true_grid = true_H.reshape(NY, NX).numpy()
    pred_grid = pred_H.reshape(NY, NX).numpy()
    error_grid = absolute_error.reshape(NY, NX).numpy()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        constrained_layout=True,
    )
    panels = (
        (true_grid, f"True {panel_prefix} H", "Blues", depth_vmax),
        (pred_grid, f"Predicted {panel_prefix} H", "Blues", depth_vmax),
        (
            error_grid,
            f"|Predicted - true| {panel_prefix} H",
            "magma",
            error_vmax,
        ),
    )
    for axis, (grid, title, color_map, vmax) in zip(
        axes,
        panels,
        strict=True,
    ):
        image = axis.imshow(
            grid,
            origin="upper",
            aspect="equal",
            cmap=color_map,
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("Grid x index")
        axis.set_ylabel("Grid y index")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(figure_title)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_max_H_timeseries(
    timestep_metrics: list[dict],
    output_path: Path,
    event_id: str,
    mode: str,
    dpi: int,
) -> None:
    """Plot true and predicted maximum water depth over target time."""
    minutes = [int(row["target_minutes"]) for row in timestep_metrics]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(
        minutes,
        [float(row["true_H_max"]) for row in timestep_metrics],
        marker="o",
        label="True max H",
    )
    axis.plot(
        minutes,
        [float(row["pred_H_max"]) for row in timestep_metrics],
        marker="o",
        label="Predicted max H",
    )
    axis.set(
        title=f"{event_id} {mode}: maximum H",
        xlabel="Target time (minutes)",
        ylabel="Maximum H (m)",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_RMSE_timeseries(
    timestep_metrics: list[dict],
    output_path: Path,
    event_id: str,
    mode: str,
    dpi: int,
) -> None:
    """Plot H and Qmag RMSE over target time."""
    minutes = [int(row["target_minutes"]) for row in timestep_metrics]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(
        minutes,
        [float(row["H_RMSE"]) for row in timestep_metrics],
        marker="o",
        label="H RMSE",
    )
    axis.plot(
        minutes,
        [float(row["Qmag_RMSE"]) for row in timestep_metrics],
        marker="o",
        label="Qmag RMSE",
    )
    axis.set(
        title=f"{event_id} {mode}: RMSE",
        xlabel="Target time (minutes)",
        ylabel="RMSE",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_CSI_timeseries(
    timestep_metrics: list[dict],
    output_path: Path,
    event_id: str,
    mode: str,
    dpi: int,
) -> None:
    """Plot CSI at all requested depth thresholds over target time."""
    minutes = [int(row["target_minutes"]) for row in timestep_metrics]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, threshold in CSI_THRESHOLDS.items():
        values = [
            math.nan if row[label] is None else float(row[label])
            for row in timestep_metrics
        ]
        axis.plot(
            minutes,
            values,
            marker="o",
            label=f"CSI {threshold:.2f} m",
        )
    axis.set(
        title=f"{event_id} {mode}: critical success index",
        xlabel="Target time (minutes)",
        ylabel="CSI",
        ylim=(0.0, 1.05),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def create_event_figures(
    prediction_data: dict,
    timestep_metrics: list[dict],
    checkpoint_path: Path,
    maps_dir: Path,
    timeseries_dir: Path,
    requested_depth_vmax: float,
    requested_error_vmax: float,
    dpi: int,
) -> tuple[list[Path], list[Path]]:
    """Create both maps and all three time-series figures for an event/mode."""
    event_id = prediction_data["event_id"]
    mode = prediction_data["mode"]
    figure_title = diagnostic_figure_title(
        event_id,
        mode,
        checkpoint_path,
        prediction_data["start_time_index"],
        prediction_data["end_time_index"],
    )
    map_paths = [
        maps_dir / f"{event_id}_{mode}_max_depth_comparison.png",
        maps_dir / f"{event_id}_{mode}_final_depth_comparison.png",
    ]
    plot_depth_comparison(
        prediction_data["true_H"].max(dim=0).values,
        prediction_data["pred_H"].max(dim=0).values,
        map_paths[0],
        "maximum",
        figure_title,
        requested_depth_vmax,
        requested_error_vmax,
        dpi,
    )
    plot_depth_comparison(
        prediction_data["true_H"][-1],
        prediction_data["pred_H"][-1],
        map_paths[1],
        "final",
        figure_title,
        requested_depth_vmax,
        requested_error_vmax,
        dpi,
    )

    timeseries_paths = [
        timeseries_dir / f"{event_id}_{mode}_maxH_timeseries.png",
        timeseries_dir / f"{event_id}_{mode}_rmse_timeseries.png",
        timeseries_dir / f"{event_id}_{mode}_csi_timeseries.png",
    ]
    plot_max_H_timeseries(
        timestep_metrics,
        timeseries_paths[0],
        event_id,
        mode,
        dpi,
    )
    plot_RMSE_timeseries(
        timestep_metrics,
        timeseries_paths[1],
        event_id,
        mode,
        dpi,
    )
    plot_CSI_timeseries(
        timestep_metrics,
        timeseries_paths[2],
        event_id,
        mode,
        dpi,
    )
    return map_paths, timeseries_paths


def plot_grouped_event_metric(
    event_summaries: list[dict],
    selected_event_ids: list[str],
    modes: list[str],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    reference_value: float | None = None,
) -> None:
    """Create a grouped bar chart for one event-level metric."""
    summary_by_key = {
        (row["event_id"], row["mode"]): row for row in event_summaries
    }
    positions = list(range(len(selected_event_ids)))
    bar_width = 0.8 / len(modes)
    figure, axis = plt.subplots(
        figsize=(max(8, 1.4 * len(selected_event_ids)), 5),
        constrained_layout=True,
    )
    for mode_index, mode in enumerate(modes):
        offset = (mode_index - (len(modes) - 1) / 2.0) * bar_width
        values = [
            summary_by_key[(event_id, mode)][metric]
            for event_id in selected_event_ids
        ]
        axis.bar(
            [position + offset for position in positions],
            [math.nan if value is None else float(value) for value in values],
            width=bar_width,
            label=mode,
        )
    if reference_value is not None:
        axis.axhline(
            reference_value,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"reference = {reference_value:g}",
        )
    axis.set(
        title=title,
        xlabel="Validation event",
        ylabel=ylabel,
        xticks=positions,
        xticklabels=selected_event_ids,
    )
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_autoregressive_CSI_summary(
    event_summaries: list[dict],
    selected_event_ids: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Create the selected-event autoregressive mean-CSI bar chart."""
    autoregressive = {
        row["event_id"]: row
        for row in event_summaries
        if row["mode"] == "autoregressive"
    }
    positions = list(range(len(selected_event_ids)))
    figure, axis = plt.subplots(
        figsize=(max(8, 1.4 * len(selected_event_ids)), 5),
        constrained_layout=True,
    )
    if autoregressive:
        csi_fields = (
            ("mean_CSI_0p05", 0.05),
            ("mean_CSI_0p10", 0.10),
            ("mean_CSI_0p20", 0.20),
        )
        bar_width = 0.8 / len(csi_fields)
        for field_index, (field, threshold) in enumerate(csi_fields):
            offset = (
                field_index - (len(csi_fields) - 1) / 2.0
            ) * bar_width
            values = [
                autoregressive[event_id][field]
                for event_id in selected_event_ids
            ]
            axis.bar(
                [position + offset for position in positions],
                [
                    math.nan if value is None else float(value)
                    for value in values
                ],
                width=bar_width,
                label=f"CSI {threshold:.2f} m",
            )
    else:
        axis.text(
            0.5,
            0.5,
            "Autoregressive mode was not selected",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(
        title="Selected validation events: autoregressive mean CSI",
        xlabel="Validation event",
        ylabel="Mean CSI",
        ylim=(0.0, 1.05),
        xticks=positions,
        xticklabels=selected_event_ids,
    )
    axis.grid(True, axis="y", alpha=0.3)
    if autoregressive:
        axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def create_summary_figures(
    event_summaries: list[dict],
    selected_event_ids: list[str],
    modes: list[str],
    summary_dir: Path,
    dpi: int,
) -> list[Path]:
    """Create all three requested cross-event summary figures."""
    paths = [
        summary_dir / "selected_events_H_peak_ratio.png",
        summary_dir / "selected_events_mean_H_RMSE.png",
        summary_dir / "selected_events_autoregressive_CSI.png",
    ]
    plot_grouped_event_metric(
        event_summaries,
        selected_event_ids,
        modes,
        "H_peak_ratio",
        "Predicted / true maximum H",
        "Selected validation events: H peak ratio",
        paths[0],
        dpi,
        reference_value=1.0,
    )
    plot_grouped_event_metric(
        event_summaries,
        selected_event_ids,
        modes,
        "mean_H_RMSE",
        "Mean H RMSE (m)",
        "Selected validation events: mean H RMSE",
        paths[1],
        dpi,
    )
    plot_autoregressive_CSI_summary(
        event_summaries,
        selected_event_ids,
        paths[2],
        dpi,
    )
    return paths


def optional_float_text(value: float | None) -> str:
    """Format an optional metric for concise console output."""
    if value is None:
        return "undefined"
    return f"{value:.8g}"


def print_event_diagnostics(summary: dict) -> None:
    """Print the requested concise diagnostics for one event and mode."""
    print(f"\nEvent {summary['event_id']} | mode {summary['mode']}")
    print("max_true_H_over_eval:", summary["max_true_H_over_eval"])
    print("max_pred_H_over_eval:", summary["max_pred_H_over_eval"])
    print(
        "H_peak_ratio:",
        optional_float_text(summary["H_peak_ratio"]),
    )
    print("mean_H_RMSE:", summary["mean_H_RMSE"])
    print("mean_combined_MSE:", summary["mean_combined_MSE"])
    print("Mean CSI values:")
    for label in CSI_THRESHOLDS:
        print(
            f"  {label}: ",
            optional_float_text(summary[f"mean_{label}"]),
        )


def prepare_output_dir(
    output_dir: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    event_metrics_csv: Path,
    overwrite: bool,
) -> None:
    """Create or safely recreate only the diagnostic output directory."""
    output_exists = output_dir.exists() or output_dir.is_symlink()
    if output_exists and output_dir.is_symlink():
        raise ValueError(f"Refusing to use symlink --output-dir: {output_dir}")
    if output_exists and not output_dir.is_dir():
        raise ValueError(f"--output-dir is not a directory: {output_dir}")
    if output_exists and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite nonempty Milestone 14B-4 output "
            f"directory without --overwrite: {output_dir}"
        )
    if output_exists and overwrite:
        resolved_output = output_dir.resolve()
        resolved_dataset = dataset_root.resolve()
        resolved_repo = Path(__file__).resolve().parents[1]
        resolved_swegnn_repo = SWEGNN_REPO.resolve()
        resolved_checkpoint_dir = checkpoint_path.resolve().parent
        resolved_metrics_dir = event_metrics_csv.resolve().parent
        resolved_cwd = Path.cwd().resolve()
        prohibited = {
            Path("/").resolve(),
            Path.home().resolve(),
            resolved_dataset,
            resolved_repo,
            resolved_swegnn_repo,
            resolved_checkpoint_dir,
            resolved_metrics_dir,
            resolved_cwd,
        }
        if resolved_output in prohibited:
            raise ValueError(f"Refusing unsafe --overwrite target: {resolved_output}")
        protected_targets = (
            resolved_dataset,
            resolved_repo,
            resolved_swegnn_repo,
            resolved_checkpoint_dir,
            resolved_metrics_dir,
            resolved_cwd,
            DEFAULT_OUTPUT_DIR.resolve(),
        )
        if any(
            resolved_output in target.parents
            for target in protected_targets
        ):
            raise ValueError(
                f"Refusing to delete broad parent directory: {resolved_output}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()
    checkpoint_path = args.checkpoint.expanduser()
    event_metrics_csv = args.event_metrics_csv.expanduser()
    output_dir = args.output_dir.expanduser()
    device = select_device(args.device, args.allow_cpu)
    modes = parse_modes(args.modes)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not SWEGNN_REPO.is_dir():
        raise FileNotFoundError(f"Original SWE-GNN repo not found: {SWEGNN_REPO}")
    if args.start_time_index < 0:
        raise ValueError("--start-time-index must be nonnegative")
    if args.end_time_index >= NT - 1:
        raise ValueError(
            f"--end-time-index must be less than {NT - 1}; "
            f"got {args.end_time_index}"
        )
    if args.start_time_index > args.end_time_index:
        raise ValueError("--start-time-index must not exceed --end-time-index")
    if not math.isfinite(args.rainfall_scale):
        raise ValueError("--rainfall-scale must be finite")
    if args.hid_features <= 0:
        raise ValueError("--hid-features must be greater than zero")
    if args.K <= 0:
        raise ValueError("--K must be greater than zero")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")
    if (
        not math.isfinite(args.max_depth_vmax)
        or args.max_depth_vmax < 0.0
    ):
        raise ValueError("--max-depth-vmax must be finite and nonnegative")
    if not math.isfinite(args.error_vmax) or args.error_vmax < 0.0:
        raise ValueError("--error-vmax must be finite and nonnegative")

    timestep_metrics_csv = (
        output_dir / "milestone14b4_timestep_metrics.csv"
    )
    event_summary_csv = output_dir / "milestone14b4_event_summary.csv"
    summary_json = output_dir / "milestone14b4_summary.json"
    predictions_dir = output_dir / "predictions"
    figures_dir = output_dir / "figures"
    maps_dir = figures_dir / "maps"
    timeseries_dir = figures_dir / "timeseries"
    figure_summary_dir = figures_dir / "summary"

    manifest = load_manifest(dataset_root)
    valid_event_ids = derive_split_event_ids(
        dataset_root,
        manifest,
        "valid",
    )
    if len(valid_event_ids) != EXPECTED_SPLIT_COUNTS["valid"]:
        raise ValueError(
            f"Expected {EXPECTED_SPLIT_COUNTS['valid']} validation events, "
            f"found {len(valid_event_ids)}"
        )
    if "R019" in valid_event_ids:
        raise ValueError("R019 must not be present in validation event IDs")
    selected_event_ids, selection_method = resolve_selected_event_ids(
        args.event_ids,
        event_metrics_csv,
        valid_event_ids,
    )
    if any(event_id not in valid_event_ids for event_id in selected_event_ids):
        raise ValueError("Every selected event must be in the validation split")
    if "R019" in selected_event_ids:
        raise ValueError("R019 must not be selected")

    checkpoint, checkpoint_metadata = load_and_validate_checkpoint(
        checkpoint_path,
        device,
        args,
    )
    checkpoint_rainfall_scale = float(
        checkpoint_metadata["checkpoint_rainfall_scale"]
    )
    if args.rainfall_scale != checkpoint_rainfall_scale:
        print(
            "WARNING: diagnostic rainfall scale differs from checkpoint: "
            f"{args.rainfall_scale} vs {checkpoint_rainfall_scale}"
        )

    time_indices = list(
        range(args.start_time_index, args.end_time_index + 1)
    )
    target_time_indices = [time_index + 1 for time_index in time_indices]
    target_minutes = [
        time_index * TIME_STEP_MINUTES for time_index in target_time_indices
    ]

    print("=== Stockbridge SWE-GNN milestone 14B-4 diagnostics ===")
    print("Dataset root:", dataset_root)
    print("Checkpoint path:", checkpoint_path)
    print("Output directory:", output_dir)
    print("Selected device:", device)
    print("Selected events:", selected_event_ids)
    print("Selection method:", selection_method)
    print("Modes:", modes)
    print(
        "Evaluation range: "
        f"T{args.start_time_index}->T{args.start_time_index + 1} through "
        f"T{args.end_time_index}->T{args.end_time_index + 1}"
    )
    print("Target minutes:", target_minutes)
    print("Clamp output:", args.clamp_output)
    print("Clamp feedback:", args.clamp_feedback)
    print("Checkpoint type:", checkpoint_metadata["checkpoint_type"])
    print(
        "Checkpoint best epoch:",
        checkpoint_metadata["checkpoint_best_epoch"],
    )
    print(
        "Checkpoint best loss:",
        checkpoint_metadata["checkpoint_best_loss"],
    )
    print("Checkpoint rainfall scale:", checkpoint_rainfall_scale)
    print(
        "Checkpoint time range: "
        f"T{checkpoint_metadata['checkpoint_start_time_index']} through "
        f"T{checkpoint_metadata['checkpoint_end_time_index']}"
    )

    prepare_output_dir(
        output_dir,
        dataset_root,
        checkpoint_path,
        event_metrics_csv,
        args.overwrite,
    )
    for directory in (
        predictions_dir,
        maps_dir,
        timeseries_dir,
        figure_summary_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    static_graph = load_static_graph(dataset_root)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    static_device_graph = static_graph_to_device(static_graph, device)
    prepared_valid_events = prepare_dynamic_events(
        dataset_root,
        "valid",
        selected_event_ids,
        device,
    )

    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)
    from models.gnn import GNN

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = GNN(
        node_features=7,
        edge_features=3,
        type_GNN="SWEGNN",
        hid_features=args.hid_features,
        K=args.K,
        gnn_activation="tanh",
        dropout=0.0,
        mlp_layers=2,
        mlp_activation="prelu",
        seed=args.seed,
        with_filter_matrix=True,
        with_gradient=True,
        with_WL=True,
        previous_t=1,
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()

    all_timestep_metrics = []
    all_event_summaries = []
    prediction_paths = []
    map_paths = []
    timeseries_paths = []
    for event in prepared_valid_events:
        for mode in modes:
            (
                timestep_metrics,
                event_summary,
                prediction_data,
            ) = evaluate_event_mode(
                model,
                static_device_graph,
                event,
                time_indices,
                args.rainfall_scale,
                mode,
                args.clamp_output,
                args.clamp_feedback,
            )
            prediction_path = (
                predictions_dir / mode / f"{event['event_id']}.pt"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prediction_data, prediction_path)
            event_map_paths, event_timeseries_paths = create_event_figures(
                prediction_data,
                timestep_metrics,
                checkpoint_path,
                maps_dir,
                timeseries_dir,
                args.max_depth_vmax,
                args.error_vmax,
                args.dpi,
            )
            all_timestep_metrics.extend(timestep_metrics)
            all_event_summaries.append(event_summary)
            prediction_paths.append(prediction_path)
            map_paths.extend(event_map_paths)
            timeseries_paths.extend(event_timeseries_paths)
            print_event_diagnostics(event_summary)

    write_metrics_csv(
        timestep_metrics_csv,
        TIMESTEP_METRICS_FIELDNAMES,
        all_timestep_metrics,
    )
    write_metrics_csv(
        event_summary_csv,
        EVENT_SUMMARY_FIELDNAMES,
        all_event_summaries,
    )
    summary_figure_paths = create_summary_figures(
        all_event_summaries,
        selected_event_ids,
        modes,
        figure_summary_dir,
        args.dpi,
    )

    expected_evaluations = len(selected_event_ids) * len(modes)
    if len(prediction_paths) != expected_evaluations:
        raise RuntimeError("Unexpected number of saved prediction files")
    if len(map_paths) != 2 * expected_evaluations:
        raise RuntimeError("Unexpected number of map figures")
    if len(timeseries_paths) != 3 * expected_evaluations:
        raise RuntimeError("Unexpected number of time-series figures")
    for path in (
        prediction_paths
        + map_paths
        + timeseries_paths
        + summary_figure_paths
    ):
        if not path.is_file():
            raise RuntimeError(f"Expected diagnostic output was not saved: {path}")

    cuda_device_name = None
    max_memory_allocated_mb = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)
        max_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (
            1024**2
        )
    total_elapsed_seconds = time.perf_counter() - run_start
    summary = {
        "script_name": Path(__file__).name,
        "dataset_root": str(dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "event_metrics_csv": str(event_metrics_csv),
        "output_dir": str(output_dir),
        "selected_event_ids": selected_event_ids,
        "selection_method": selection_method,
        "valid_event_ids_count": len(valid_event_ids),
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "rainfall_scale": args.rainfall_scale,
        "checkpoint_rainfall_scale": checkpoint_rainfall_scale,
        "start_time_index": args.start_time_index,
        "end_time_index": args.end_time_index,
        "target_time_indices": target_time_indices,
        "target_minutes": target_minutes,
        "modes": modes,
        "clamp_output": args.clamp_output,
        "clamp_feedback": args.clamp_feedback,
        "checkpoint_type": checkpoint_metadata["checkpoint_type"],
        "checkpoint_best_epoch": checkpoint_metadata[
            "checkpoint_best_epoch"
        ],
        "checkpoint_best_loss": checkpoint_metadata[
            "checkpoint_best_loss"
        ],
        "hid_features": args.hid_features,
        "K": args.K,
        "seed": args.seed,
        "num_saved_prediction_files": len(prediction_paths),
        "num_map_figures": len(map_paths),
        "num_timeseries_figures": len(timeseries_paths),
        "num_summary_figures": len(summary_figure_paths),
        "timestep_metrics_csv": str(timestep_metrics_csv),
        "event_summary_csv": str(event_summary_csv),
        "predictions_dir": str(predictions_dir),
        "figures_dir": str(figures_dir),
        "device": str(device),
        "total_elapsed_seconds": total_elapsed_seconds,
    }
    if device.type == "cuda":
        summary["cuda_device_name"] = cuda_device_name
        summary["max_memory_allocated_mb"] = max_memory_allocated_mb
    with summary_json.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)

    print("\nOutput paths:")
    print(timestep_metrics_csv)
    print(event_summary_csv)
    print(summary_json)
    print(predictions_dir)
    print(figures_dir)
    print("Saved prediction files:", len(prediction_paths))
    print("Map figures:", len(map_paths))
    print("Time-series figures:", len(timeseries_paths))
    print("Summary figures:", len(summary_figure_paths))
    print(
        "Total figures:",
        len(map_paths) + len(timeseries_paths) + len(summary_figure_paths),
    )
    if device.type == "cuda":
        print("\nCUDA diagnostics:")
        print("device name:", cuda_device_name)
        print(f"max memory allocated: {max_memory_allocated_mb:.2f} MB")
    print()
    print(
        "Stockbridge SWE-GNN milestone 14B-4 validation visual diagnostic "
        "passed."
    )


if __name__ == "__main__":
    main()
