import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data


DATASET_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_swegnn_full_v01"
)
SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14a_full_train_smoke_v01"
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

HISTORY_FIELDNAMES = [
    "epoch",
    "train_loss_mean",
    "train_loss_min",
    "train_loss_max",
    "valid_loss_mean",
    "valid_H_RMSE_mean",
    "valid_H_MAE_mean",
    "valid_Qmag_RMSE_mean",
    "valid_Qmag_MAE_mean",
    "valid_combined_MSE_mean",
    "valid_H_peak_ratio_mean",
    "best_valid_loss_so_far",
    "is_best",
    "epoch_elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for milestone 14A."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a short one-step teacher-forced SWE-GNN training smoke test "
            "on the full Stockbridge train/validation split."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the full-graph training smoke test to run on CPU.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-time-index", type=int, default=2)
    parser.add_argument("--end-time-index", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--max-valid-events", type=int, default=0)
    return parser.parse_args()


def select_device(requested_device: str, allow_cpu: bool) -> torch.device:
    """Resolve the device and protect against accidental CPU training."""
    if requested_device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = requested_device

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    if device_type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Refusing to run full-graph milestone 14A "
            "training on CPU unless --allow-cpu is set."
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


def apply_event_limit(
    event_ids: list[str],
    limit: int,
    split: str,
) -> list[str]:
    """Apply a positive debugging limit, leaving zero as all events."""
    if limit < 0:
        raise ValueError(f"--max-{split}-events must be nonnegative")
    if limit == 0:
        return event_ids
    if limit > len(event_ids):
        raise ValueError(
            f"--max-{split}-events={limit} exceeds the available "
            f"{len(event_ids)} events"
        )
    return event_ids[:limit]


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


def summarize_losses(losses: list[float]) -> tuple[float, float, float]:
    """Return mean, minimum, and maximum for a non-empty loss list."""
    if not losses:
        raise ValueError("Cannot summarize an empty loss list")
    return sum(losses) / len(losses), min(losses), max(losses)


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


def evaluate_validation(
    model: torch.nn.Module,
    static_device_graph: dict,
    prepared_valid_events: list[dict],
    time_indices: list[int],
    rainfall_scale: float,
    criterion: torch.nn.Module,
) -> dict[str, float | int | None]:
    """Run teacher-forced validation and aggregate pair/event metrics."""
    losses = []
    H_RMSE_values = []
    H_MAE_values = []
    Qmag_RMSE_values = []
    Qmag_MAE_values = []
    combined_MSE_values = []
    event_H_peak_ratios = []

    model.eval()
    with torch.inference_mode():
        for event in prepared_valid_events:
            max_pred_H_over_eval = -math.inf
            max_true_H_over_eval = -math.inf
            for t in time_indices:
                sample = build_device_sample_from_prepared(
                    static_device_graph,
                    event,
                    t,
                    rainfall_scale,
                )
                context = f"validation for {event['event_id']} at T{t}"
                pred = extract_prediction(model(sample), context)
                if tuple(pred.shape) != (NUM_NODES, 2):
                    raise ValueError(
                        f"Expected prediction shape {(NUM_NODES, 2)} during "
                        f"{context}, got {tuple(pred.shape)}"
                    )
                assert_finite(f"{event['event_id']} validation prediction", pred)

                H_error = pred[:, 0] - sample.y[:, 0]
                Qmag_error = pred[:, 1] - sample.y[:, 1]
                H_MSE = torch.mean(H_error.square())
                Qmag_MSE = torch.mean(Qmag_error.square())
                combined_MSE = torch.mean((pred - sample.y).square())
                loss = criterion(pred, sample.y)
                metric_tensors = {
                    "H_MSE": H_MSE,
                    "Qmag_MSE": Qmag_MSE,
                    "combined_MSE": combined_MSE,
                    "loss": loss,
                }
                for name, value in metric_tensors.items():
                    assert_finite(f"{event['event_id']} validation {name}", value)

                losses.append(loss.item())
                H_RMSE_values.append(torch.sqrt(H_MSE).item())
                H_MAE_values.append(torch.mean(torch.abs(H_error)).item())
                Qmag_RMSE_values.append(torch.sqrt(Qmag_MSE).item())
                Qmag_MAE_values.append(torch.mean(torch.abs(Qmag_error)).item())
                combined_MSE_values.append(combined_MSE.item())
                max_pred_H_over_eval = max(
                    max_pred_H_over_eval,
                    pred[:, 0].max().item(),
                )
                max_true_H_over_eval = max(
                    max_true_H_over_eval,
                    sample.y[:, 0].max().item(),
                )
            event_H_peak_ratios.append(
                safe_peak_ratio(max_pred_H_over_eval, max_true_H_over_eval)
            )

    valid_loss_mean = sum(losses) / len(losses)
    valid_combined_MSE_mean = sum(combined_MSE_values) / len(
        combined_MSE_values
    )
    summary = {
        "num_events": len(prepared_valid_events),
        "num_pairs": len(losses),
        "valid_loss_mean": valid_loss_mean,
        "valid_H_RMSE_mean": sum(H_RMSE_values) / len(H_RMSE_values),
        "valid_H_MAE_mean": sum(H_MAE_values) / len(H_MAE_values),
        "valid_Qmag_RMSE_mean": (
            sum(Qmag_RMSE_values) / len(Qmag_RMSE_values)
        ),
        "valid_Qmag_MAE_mean": sum(Qmag_MAE_values) / len(Qmag_MAE_values),
        "valid_combined_MSE_mean": valid_combined_MSE_mean,
        "valid_H_peak_ratio_mean": mean_optional(event_H_peak_ratios),
    }
    for name, value in summary.items():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"Non-finite validation summary value {name}={value}")
    return summary


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    static_device_graph: dict,
    prepared_train_events: list[dict],
    training_pairs: list[tuple[int, int]],
    rainfall_scale: float,
    shuffle: bool,
) -> tuple[float, float, float]:
    """Run one epoch of one-step teacher-forced training."""
    epoch_pairs = training_pairs.copy()
    if shuffle:
        random.shuffle(epoch_pairs)

    model.train()
    losses = []
    for event_index, t in epoch_pairs:
        event = prepared_train_events[event_index]
        sample = build_device_sample_from_prepared(
            static_device_graph,
            event,
            t,
            rainfall_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        context = f"training for {event['event_id']} at T{t}"
        pred = extract_prediction(model(sample), context)
        if tuple(pred.shape) != (NUM_NODES, 2):
            raise ValueError(
                f"Expected prediction shape {(NUM_NODES, 2)} during "
                f"{context}, got {tuple(pred.shape)}"
            )
        assert_finite(f"{event['event_id']} training prediction", pred)
        loss = criterion(pred, sample.y)
        assert_finite(f"{event['event_id']} training loss", loss)
        loss.backward()

        parameters_with_gradients = [
            parameter
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not parameters_with_gradients:
            raise RuntimeError(
                f"No model parameter received a gradient during {context}"
            )
        if not all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in parameters_with_gradients
        ):
            raise ValueError(f"Non-finite model gradient during {context}")
        optimizer.step()
        losses.append(loss.item())
    return summarize_losses(losses)


def build_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_valid_loss: float,
    args: argparse.Namespace,
    train_event_ids: list[str],
    valid_event_ids: list[str],
    model_config: dict,
    train_pairs_per_epoch: int,
    valid_pairs_per_epoch: int,
    valid_summary: dict,
) -> dict:
    """Build a self-describing Milestone 14A checkpoint dictionary."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_valid_loss": best_valid_loss,
        "args": vars(args).copy(),
        "train_event_ids": train_event_ids,
        "valid_event_ids": valid_event_ids,
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "model_config": model_config,
        "rainfall_scale": args.rainfall_scale,
        "start_time_index": args.start_time_index,
        "end_time_index": args.end_time_index,
        "train_pairs_per_epoch": train_pairs_per_epoch,
        "valid_pairs_per_epoch": valid_pairs_per_epoch,
        "valid_summary": valid_summary,
    }


def optional_float_text(value: float | None) -> str:
    """Format an optional metric for concise console diagnostics."""
    if value is None:
        return "undefined"
    return f"{value:.8g}"


def prepare_output_dir(
    output_dir: Path,
    dataset_root: Path,
    overwrite: bool,
) -> None:
    """Create the output directory, safely recreating only it when requested."""
    output_exists = output_dir.exists() or output_dir.is_symlink()
    if output_exists and output_dir.is_symlink():
        raise ValueError(f"Refusing to use symlink --output-dir: {output_dir}")
    if output_exists and not output_dir.is_dir():
        raise ValueError(f"--output-dir is not a directory: {output_dir}")
    if output_exists and overwrite:
        resolved_output = output_dir.resolve()
        resolved_dataset = dataset_root.resolve()
        resolved_cwd = Path.cwd().resolve()
        resolved_default_output = DEFAULT_OUTPUT_DIR.resolve()
        prohibited = {
            Path("/").resolve(),
            Path.home().resolve(),
            resolved_dataset,
            resolved_cwd,
        }
        if resolved_output in prohibited:
            raise ValueError(f"Refusing unsafe --overwrite target: {resolved_output}")
        if (
            resolved_output in resolved_dataset.parents
            or resolved_output in resolved_cwd.parents
            or resolved_output in resolved_default_output.parents
        ):
            raise ValueError(
                f"Refusing to delete broad parent directory: {resolved_output}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()
    output_dir = args.output_dir.expanduser()
    device = select_device(args.device, args.allow_cpu)

    history_csv = output_dir / "milestone14a_history.csv"
    best_checkpoint_path = output_dir / "milestone14a_best_checkpoint.pt"
    final_checkpoint_path = output_dir / "milestone14a_final_checkpoint.pt"
    summary_json = output_dir / "milestone14a_summary.json"
    output_paths = [
        history_csv,
        best_checkpoint_path,
        final_checkpoint_path,
        summary_json,
    ]

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
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
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("--lr must be finite and greater than zero")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be finite and nonnegative")
    if not math.isfinite(args.rainfall_scale):
        raise ValueError("--rainfall-scale must be finite")
    if args.hid_features <= 0:
        raise ValueError("--hid-features must be greater than zero")
    if args.K <= 0:
        raise ValueError("--K must be greater than zero")
    if args.max_train_events < 0:
        raise ValueError("--max-train-events must be nonnegative")
    if args.max_valid_events < 0:
        raise ValueError("--max-valid-events must be nonnegative")
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing milestone 14A outputs without "
            f"--overwrite:\n{formatted}"
        )

    static_graph = load_static_graph(dataset_root)
    manifest = load_manifest(dataset_root)
    full_train_event_ids = derive_split_event_ids(
        dataset_root,
        manifest,
        "train",
    )
    full_valid_event_ids = derive_split_event_ids(
        dataset_root,
        manifest,
        "valid",
    )
    if "R019" in full_train_event_ids or "R019" in full_valid_event_ids:
        raise ValueError("R019 must not be present in train or valid event IDs")
    overlap = sorted(
        set(full_train_event_ids).intersection(full_valid_event_ids),
        key=natural_event_sort_key,
    )
    if overlap:
        raise ValueError(f"Train and valid event IDs overlap: {overlap}")

    train_event_ids = apply_event_limit(
        full_train_event_ids,
        args.max_train_events,
        "train",
    )
    valid_event_ids = apply_event_limit(
        full_valid_event_ids,
        args.max_valid_events,
        "valid",
    )
    debugging_subset = bool(
        args.max_train_events > 0 or args.max_valid_events > 0
    )
    if debugging_subset:
        print(
            "WARNING: --max-train-events/--max-valid-events selected a "
            "debugging subset; this is not the default full 70/15 run."
        )

    time_indices = list(
        range(args.start_time_index, args.end_time_index + 1)
    )
    training_pairs = [
        (event_index, t)
        for event_index in range(len(train_event_ids))
        for t in time_indices
    ]
    train_pairs_per_epoch = len(training_pairs)
    valid_pairs_per_epoch = len(valid_event_ids) * len(time_indices)
    expected_train_pairs = len(train_event_ids) * len(time_indices)
    expected_valid_pairs = len(valid_event_ids) * len(time_indices)
    assert train_pairs_per_epoch == expected_train_pairs
    assert valid_pairs_per_epoch == expected_valid_pairs

    print("=== Stockbridge SWE-GNN milestone 14A full training smoke ===")
    print("Dataset root:", dataset_root)
    print("Output directory:", output_dir)
    print("Selected device:", device)
    print("Train events count:", len(train_event_ids))
    print("Valid events count:", len(valid_event_ids))
    print("Train pairs per epoch:", train_pairs_per_epoch)
    print("Valid pairs per epoch:", valid_pairs_per_epoch)
    print(
        "Time range: "
        f"T{args.start_time_index}->T{args.start_time_index + 1} through "
        f"T{args.end_time_index}->T{args.end_time_index + 1}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    static_device_graph = static_graph_to_device(static_graph, device)
    prepared_train_events = prepare_dynamic_events(
        dataset_root,
        "train",
        train_event_ids,
        device,
    )
    prepared_valid_events = prepare_dynamic_events(
        dataset_root,
        "valid",
        valid_event_ids,
        device,
    )

    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)
    from models.gnn import GNN

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_config = {
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
        "device": str(device),
    }
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.MSELoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    first_trainable_parameter_name, first_trainable_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    first_parameter_before = first_trainable_parameter.detach().clone()

    print("Model config:", model_config)
    print("Parameter count:", parameter_count)
    initial_valid_summary = evaluate_validation(
        model,
        static_device_graph,
        prepared_valid_events,
        time_indices,
        args.rainfall_scale,
        criterion,
    )
    print("Initial validation:", initial_valid_summary)

    prepare_output_dir(output_dir, dataset_root, args.overwrite)
    best_valid_loss = math.inf
    best_epoch = 0

    with history_csv.open("w", newline="") as history_file:
        history_writer = csv.DictWriter(
            history_file,
            fieldnames=HISTORY_FIELDNAMES,
        )
        history_writer.writeheader()
        history_file.flush()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss_summary = train_one_epoch(
                model,
                optimizer,
                criterion,
                static_device_graph,
                prepared_train_events,
                training_pairs,
                args.rainfall_scale,
                args.shuffle,
            )
            valid_summary = evaluate_validation(
                model,
                static_device_graph,
                prepared_valid_events,
                time_indices,
                args.rainfall_scale,
                criterion,
            )
            valid_loss_mean = float(valid_summary["valid_loss_mean"])
            is_best = valid_loss_mean < best_valid_loss
            if is_best:
                best_valid_loss = valid_loss_mean
                best_epoch = epoch
                torch.save(
                    build_checkpoint(
                        epoch,
                        model,
                        optimizer,
                        best_valid_loss,
                        args,
                        train_event_ids,
                        valid_event_ids,
                        model_config,
                        train_pairs_per_epoch,
                        valid_pairs_per_epoch,
                        valid_summary,
                    ),
                    best_checkpoint_path,
                )
            epoch_elapsed_seconds = time.perf_counter() - epoch_start

            history_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss_mean": train_loss_summary[0],
                    "train_loss_min": train_loss_summary[1],
                    "train_loss_max": train_loss_summary[2],
                    "valid_loss_mean": valid_summary["valid_loss_mean"],
                    "valid_H_RMSE_mean": valid_summary["valid_H_RMSE_mean"],
                    "valid_H_MAE_mean": valid_summary["valid_H_MAE_mean"],
                    "valid_Qmag_RMSE_mean": valid_summary[
                        "valid_Qmag_RMSE_mean"
                    ],
                    "valid_Qmag_MAE_mean": valid_summary[
                        "valid_Qmag_MAE_mean"
                    ],
                    "valid_combined_MSE_mean": valid_summary[
                        "valid_combined_MSE_mean"
                    ],
                    "valid_H_peak_ratio_mean": valid_summary[
                        "valid_H_peak_ratio_mean"
                    ],
                    "best_valid_loss_so_far": best_valid_loss,
                    "is_best": is_best,
                    "epoch_elapsed_seconds": epoch_elapsed_seconds,
                }
            )
            history_file.flush()

            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                "train loss: "
                f"mean={train_loss_summary[0]:.8g} "
                f"min={train_loss_summary[1]:.8g} "
                f"max={train_loss_summary[2]:.8g}"
            )
            print(f"valid_loss_mean: {valid_loss_mean:.8g}")
            print(
                "valid_H_RMSE_mean: "
                f"{float(valid_summary['valid_H_RMSE_mean']):.8g}"
            )
            print(
                "valid_H_peak_ratio_mean: "
                f"{optional_float_text(valid_summary['valid_H_peak_ratio_mean'])}"
            )
            print("is_best:", is_best)
            print(f"elapsed seconds: {epoch_elapsed_seconds:.2f}")

    final_valid_summary = evaluate_validation(
        model,
        static_device_graph,
        prepared_valid_events,
        time_indices,
        args.rainfall_scale,
        criterion,
    )
    if best_epoch <= 0:
        raise RuntimeError("No best checkpoint was saved")
    first_trainable_parameter_absolute_change = torch.sum(
        torch.abs(
            first_trainable_parameter.detach() - first_parameter_before
        )
    ).item()
    if (
        not math.isfinite(first_trainable_parameter_absolute_change)
        or first_trainable_parameter_absolute_change <= 0.0
    ):
        raise RuntimeError(
            "First trainable parameter change is not finite and positive: "
            f"{first_trainable_parameter_name}="
            f"{first_trainable_parameter_absolute_change}"
        )

    torch.save(
        build_checkpoint(
            args.epochs,
            model,
            optimizer,
            best_valid_loss,
            args,
            train_event_ids,
            valid_event_ids,
            model_config,
            train_pairs_per_epoch,
            valid_pairs_per_epoch,
            final_valid_summary,
        ),
        final_checkpoint_path,
    )

    cuda_device_name = None
    max_memory_allocated_mb = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)
        max_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (
            1024**2
        )
    summary = {
        "script_name": Path(__file__).name,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "train_event_ids": train_event_ids,
        "valid_event_ids": valid_event_ids,
        "num_train_events": len(train_event_ids),
        "num_valid_events": len(valid_event_ids),
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "rainfall_scale": args.rainfall_scale,
        "start_time_index": args.start_time_index,
        "end_time_index": args.end_time_index,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hid_features": args.hid_features,
        "K": args.K,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "train_pairs_per_epoch": train_pairs_per_epoch,
        "valid_pairs_per_epoch": valid_pairs_per_epoch,
        "parameter_count": parameter_count,
        "initial_valid_summary": initial_valid_summary,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "final_valid_summary": final_valid_summary,
        "first_trainable_parameter_name": first_trainable_parameter_name,
        "first_trainable_parameter_absolute_change": (
            first_trainable_parameter_absolute_change
        ),
        "device": str(device),
    }
    if device.type == "cuda":
        summary["cuda_device_name"] = cuda_device_name
        summary["max_memory_allocated_mb"] = max_memory_allocated_mb
    with summary_json.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)

    print("\nBest epoch:", best_epoch)
    print(f"Best valid loss: {best_valid_loss:.8g}")
    print("Final valid summary:", final_valid_summary)
    print("First trainable parameter:", first_trainable_parameter_name)
    print(
        "First parameter absolute change: "
        f"{first_trainable_parameter_absolute_change:.8g}"
    )
    print("\nOutput paths:")
    for path in output_paths:
        print(path)
    if device.type == "cuda":
        print("\nCUDA diagnostics:")
        print("device name:", cuda_device_name)
        print(f"max memory allocated: {max_memory_allocated_mb:.2f} MB")
    print()
    print("Stockbridge SWE-GNN milestone 14A full training smoke passed.")


if __name__ == "__main__":
    main()
