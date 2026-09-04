import argparse
import csv
import json
import math
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
DEFAULT_CHECKPOINT = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14b1_onestep_earlystop_v01"
    / "milestone14b1_best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone14b2_validation_rollout_v01"
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

EVENT_METRICS_FIELDNAMES = [
    "event_id",
    "mode",
    "num_steps",
    "start_time_index",
    "end_time_index",
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
    "pred_H_negative_min_raw",
    "pred_Qmag_negative_min_raw",
    *CSI_THRESHOLDS,
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for milestone 14B-2."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare teacher-forced and autoregressive SWE-GNN evaluation "
            "over the full Stockbridge validation split."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow full-graph validation rollout evaluation on CPU.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-time-index", type=int, default=2)
    parser.add_argument("--end-time-index", type=int, default=16)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
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
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
            "CUDA is not available. Refusing to run full-graph milestone 14B-2 "
            "evaluation on CPU unless --allow-cpu is set."
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
) -> tuple[dict, dict[str, float | int]]:
    """Load the Milestone 14B-1 checkpoint and validate its contract."""
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
        "best_valid_loss",
        "best_epoch",
    )
    require_keys(checkpoint, required_keys, "checkpoint")
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
    checkpoint_best_valid_loss = float(
        scalar_metadata(checkpoint["best_valid_loss"], "best_valid_loss")
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
        not math.isfinite(checkpoint_best_valid_loss)
        or checkpoint_best_valid_loss < 0.0
    ):
        raise ValueError(
            "checkpoint best_valid_loss must be finite and nonnegative"
        )
    if checkpoint_best_epoch <= 0:
        raise ValueError("checkpoint best_epoch must be greater than zero")

    metadata = {
        "checkpoint_rainfall_scale": checkpoint_rainfall_scale,
        "checkpoint_start_time_index": checkpoint_start_time_index,
        "checkpoint_end_time_index": checkpoint_end_time_index,
        "checkpoint_best_valid_loss": checkpoint_best_valid_loss,
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
    step_metrics: list[dict[str, float | None]],
    start_time_index: int,
    end_time_index: int,
) -> dict[str, float | int | str | None]:
    """Aggregate all evaluated timesteps for one event and mode."""
    if not step_metrics:
        raise ValueError(f"No {mode} step metrics for {event_id}")
    final_metrics = step_metrics[-1]
    max_true_H = max(float(row["true_H_max"]) for row in step_metrics)
    max_pred_H = max(float(row["pred_H_max"]) for row in step_metrics)
    summary: dict[str, float | int | str | None] = {
        "event_id": event_id,
        "mode": mode,
        "num_steps": len(step_metrics),
        "start_time_index": start_time_index,
        "end_time_index": end_time_index,
        "max_true_H_over_eval": max_true_H,
        "max_pred_H_over_eval": max_pred_H,
        "H_peak_ratio": safe_peak_ratio(max_pred_H, max_true_H),
        "mean_H_RMSE": mean_required(
            [float(row["H_RMSE"]) for row in step_metrics],
            f"{event_id} {mode} H_RMSE",
        ),
        "mean_H_MAE": mean_required(
            [float(row["H_MAE"]) for row in step_metrics],
            f"{event_id} {mode} H_MAE",
        ),
        "mean_Qmag_RMSE": mean_required(
            [float(row["Qmag_RMSE"]) for row in step_metrics],
            f"{event_id} {mode} Qmag_RMSE",
        ),
        "mean_Qmag_MAE": mean_required(
            [float(row["Qmag_MAE"]) for row in step_metrics],
            f"{event_id} {mode} Qmag_MAE",
        ),
        "mean_combined_MSE": mean_required(
            [float(row["combined_MSE"]) for row in step_metrics],
            f"{event_id} {mode} combined_MSE",
        ),
        "final_H_RMSE": float(final_metrics["H_RMSE"]),
        "final_H_MAE": float(final_metrics["H_MAE"]),
        "final_Qmag_RMSE": float(final_metrics["Qmag_RMSE"]),
        "final_Qmag_MAE": float(final_metrics["Qmag_MAE"]),
        "final_combined_MSE": float(final_metrics["combined_MSE"]),
        "pred_H_negative_min_raw": min(
            float(row["raw_pred_H_min"]) for row in step_metrics
        ),
        "pred_Qmag_negative_min_raw": min(
            float(row["raw_pred_Qmag_min"]) for row in step_metrics
        ),
    }
    for label in CSI_THRESHOLDS:
        summary[label] = mean_optional(
            [row[label] for row in step_metrics]
        )
    for name, value in summary.items():
        if name not in ("event_id", "mode") and value is not None:
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Non-finite event metric {event_id} {mode} "
                    f"{name}={value}"
                )
    return summary


def summarize_mode_metrics(
    mode: str,
    event_metrics: list[dict[str, float | int | str | None]],
    all_step_metrics: list[dict[str, float | None]],
) -> tuple[dict[str, float | int | None], dict[str, float | None]]:
    """Aggregate event metrics and pair-level CSI for one mode."""
    if not event_metrics:
        raise ValueError(f"No event metrics for {mode}")
    if not all_step_metrics:
        raise ValueError(f"No step metrics for {mode}")

    def event_mean(name: str) -> float:
        return mean_required(
            [float(row[name]) for row in event_metrics],
            f"{mode} {name}",
        )

    summary: dict[str, float | int | None] = {
        "num_events": len(event_metrics),
        "num_pairs": sum(int(row["num_steps"]) for row in event_metrics),
        "mean_H_RMSE": event_mean("mean_H_RMSE"),
        "mean_H_MAE": event_mean("mean_H_MAE"),
        "mean_Qmag_RMSE": event_mean("mean_Qmag_RMSE"),
        "mean_Qmag_MAE": event_mean("mean_Qmag_MAE"),
        "mean_combined_MSE": event_mean("mean_combined_MSE"),
        "mean_H_peak_ratio": mean_optional(
            [row["H_peak_ratio"] for row in event_metrics]
        ),
        "mean_final_H_RMSE": event_mean("final_H_RMSE"),
        "mean_final_H_MAE": event_mean("final_H_MAE"),
        "mean_final_Qmag_RMSE": event_mean("final_Qmag_RMSE"),
        "mean_final_Qmag_MAE": event_mean("final_Qmag_MAE"),
        "mean_final_combined_MSE": event_mean("final_combined_MSE"),
        "min_raw_pred_H": min(
            float(row["pred_H_negative_min_raw"])
            for row in event_metrics
        ),
        "min_raw_pred_Qmag": min(
            float(row["pred_Qmag_negative_min_raw"])
            for row in event_metrics
        ),
    }
    csi_summary = {
        label: mean_optional([row[label] for row in all_step_metrics])
        for label in CSI_THRESHOLDS
    }
    for name, value in summary.items():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"Non-finite {mode} summary {name}={value}")
    return summary, csi_summary


def save_event_predictions(
    prediction_dir: Path,
    event: dict,
    mode: str,
    start_time_index: int,
    end_time_index: int,
    pred_H_steps: list[torch.Tensor],
    pred_Qmag_steps: list[torch.Tensor],
    true_H_steps: list[torch.Tensor],
    true_Qmag_steps: list[torch.Tensor],
    clamp_output: bool,
    clamp_feedback: bool,
) -> None:
    """Save one event/mode prediction bundle on CPU."""
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_data = {
        "event_id": event["event_id"],
        "mode": mode,
        "start_time_index": start_time_index,
        "end_time_index": end_time_index,
        "pred_H": torch.stack(pred_H_steps, dim=0),
        "pred_Qmag": torch.stack(pred_Qmag_steps, dim=0),
        "true_H": torch.stack(true_H_steps, dim=0),
        "true_Qmag": torch.stack(true_Qmag_steps, dim=0),
        "rainfall_global": event["rainfall_global"].detach().cpu(),
        "clamp_output": clamp_output,
        "clamp_feedback": clamp_feedback,
    }
    expected_shape = (end_time_index - start_time_index + 1, NUM_NODES)
    for name in ("pred_H", "pred_Qmag", "true_H", "true_Qmag"):
        tensor = prediction_data[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected saved {event['event_id']} {mode} {name} shape "
                f"{expected_shape}, got {tuple(tensor.shape)}"
            )
        assert_finite(f"saved {event['event_id']} {mode} {name}", tensor)
    torch.save(
        prediction_data,
        prediction_dir / f"{event['event_id']}.pt",
    )


def evaluate_mode(
    model: torch.nn.Module,
    static_device_graph: dict,
    prepared_valid_events: list[dict],
    time_indices: list[int],
    rainfall_scale: float,
    mode: str,
    clamp_output: bool,
    clamp_feedback: bool,
    prediction_dir: Path | None,
) -> tuple[
    dict[str, float | int | None],
    list[dict[str, float | int | str | None]],
    dict[str, float | None],
]:
    """Evaluate every validation event in one inference mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown evaluation mode {mode!r}")
    if not time_indices:
        raise ValueError("Evaluation time_indices must not be empty")
    event_metrics = []
    all_step_metrics = []
    model.eval()

    with torch.inference_mode():
        for event in prepared_valid_events:
            current_H = event["H"][:, time_indices[0]]
            current_Qmag = event["Qmag"][:, time_indices[0]]
            event_step_metrics = []
            pred_H_steps = []
            pred_Qmag_steps = []
            true_H_steps = []
            true_Qmag_steps = []

            for t in time_indices:
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
                assert_finite(f"{event['event_id']} {mode} raw prediction", raw_pred)

                if clamp_output:
                    metric_pred = raw_pred.clamp_min(0.0)
                else:
                    metric_pred = raw_pred
                step_metrics = calculate_pair_metrics(
                    raw_pred,
                    metric_pred,
                    sample.y,
                )
                event_step_metrics.append(step_metrics)
                all_step_metrics.append(step_metrics)

                if prediction_dir is not None:
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

            event_metrics.append(
                summarize_event_metrics(
                    event["event_id"],
                    mode,
                    event_step_metrics,
                    time_indices[0],
                    time_indices[-1],
                )
            )
            if prediction_dir is not None:
                save_event_predictions(
                    prediction_dir,
                    event,
                    mode,
                    time_indices[0],
                    time_indices[-1],
                    pred_H_steps,
                    pred_Qmag_steps,
                    true_H_steps,
                    true_Qmag_steps,
                    clamp_output,
                    clamp_feedback,
                )

    summary, csi_summary = summarize_mode_metrics(
        mode,
        event_metrics,
        all_step_metrics,
    )
    return summary, event_metrics, csi_summary


def evaluate_teacher_forced(
    model: torch.nn.Module,
    static_device_graph: dict,
    prepared_valid_events: list[dict],
    time_indices: list[int],
    rainfall_scale: float,
    clamp_output: bool,
    clamp_feedback: bool,
    prediction_dir: Path | None = None,
) -> tuple[
    dict[str, float | int | None],
    list[dict[str, float | int | str | None]],
    dict[str, float | None],
]:
    """Evaluate one-step predictions using true state at every input time."""
    return evaluate_mode(
        model,
        static_device_graph,
        prepared_valid_events,
        time_indices,
        rainfall_scale,
        "teacher_forced",
        clamp_output,
        clamp_feedback,
        prediction_dir,
    )


def evaluate_autoregressive_rollout(
    model: torch.nn.Module,
    static_device_graph: dict,
    prepared_valid_events: list[dict],
    time_indices: list[int],
    rainfall_scale: float,
    clamp_output: bool,
    clamp_feedback: bool,
    prediction_dir: Path | None = None,
) -> tuple[
    dict[str, float | int | None],
    list[dict[str, float | int | str | None]],
    dict[str, float | None],
]:
    """Evaluate rollout predictions using model state as the next input."""
    return evaluate_mode(
        model,
        static_device_graph,
        prepared_valid_events,
        time_indices,
        rainfall_scale,
        "autoregressive",
        clamp_output,
        clamp_feedback,
        prediction_dir,
    )


def write_event_metrics(
    output_path: Path,
    event_metrics: list[dict[str, float | int | str | None]],
) -> None:
    """Write one event-level metrics CSV for one evaluation mode."""
    if not event_metrics:
        raise ValueError(f"Cannot write empty event metrics to {output_path}")
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=EVENT_METRICS_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(event_metrics)


def safe_degradation_ratio(
    numerator: float | None,
    denominator: float | None,
    label: str,
) -> float | None:
    """Calculate a finite rollout/teacher ratio when it is defined."""
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    ratio = float(numerator) / float(denominator)
    if not math.isfinite(ratio):
        raise ValueError(f"Non-finite rollout degradation ratio for {label}")
    return ratio


def optional_float_text(value: float | None) -> str:
    """Format an optional metric for concise console output."""
    if value is None:
        return "undefined"
    return f"{value:.8g}"


def print_mode_diagnostics(
    label: str,
    summary: dict[str, float | int | None],
    csi_summary: dict[str, float | None],
) -> None:
    """Print the requested aggregate diagnostics for one mode."""
    print(f"\n{label}:")
    print(f"mean_combined_MSE: {float(summary['mean_combined_MSE']):.8g}")
    print(f"mean_H_RMSE: {float(summary['mean_H_RMSE']):.8g}")
    print(
        "mean_H_peak_ratio:",
        optional_float_text(summary["mean_H_peak_ratio"]),
    )
    print("CSI by threshold:")
    for csi_label, threshold in CSI_THRESHOLDS.items():
        print(
            f"  {threshold:.2f} m ({csi_label}): "
            f"{optional_float_text(csi_summary[csi_label])}"
        )


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
    run_start = time.perf_counter()
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()
    checkpoint_path = args.checkpoint.expanduser()
    output_dir = args.output_dir.expanduser()
    device = select_device(args.device, args.allow_cpu)

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

    teacher_forced_csv = (
        output_dir / "milestone14b2_teacher_forced_event_metrics.csv"
    )
    autoregressive_csv = (
        output_dir / "milestone14b2_autoregressive_event_metrics.csv"
    )
    summary_json = output_dir / "milestone14b2_summary.json"
    predictions_root = output_dir / "predictions"
    output_paths = [teacher_forced_csv, autoregressive_csv, summary_json]
    collision_paths = output_paths.copy()
    if args.save_predictions:
        collision_paths.append(predictions_root)
    existing_outputs = [path for path in collision_paths if path.exists()]
    if existing_outputs and not args.overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing milestone 14B-2 outputs without "
            f"--overwrite:\n{formatted}"
        )
    if args.overwrite:
        resolved_output = output_dir.resolve()
        resolved_checkpoint = checkpoint_path.resolve()
        if resolved_output == resolved_checkpoint.parent:
            raise ValueError(
                "Refusing --overwrite because output_dir contains the "
                f"input checkpoint: {checkpoint_path}"
            )
        if resolved_output in resolved_checkpoint.parents:
            raise ValueError(
                "Refusing --overwrite because output_dir is an ancestor of "
                f"the input checkpoint: {output_dir}"
            )

    static_graph = load_static_graph(dataset_root)
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
            "WARNING: evaluation rainfall scale differs from checkpoint: "
            f"{args.rainfall_scale} vs {checkpoint_rainfall_scale}"
        )

    num_rollout_steps = args.end_time_index - args.start_time_index + 1
    time_indices = list(
        range(args.start_time_index, args.end_time_index + 1)
    )
    print("=== Stockbridge SWE-GNN milestone 14B-2 validation rollout ===")
    print("Dataset root:", dataset_root)
    print("Checkpoint path:", checkpoint_path)
    print("Output directory:", output_dir)
    print("Selected device:", device)
    print("Validation event count:", len(valid_event_ids))
    print(
        "Rollout range: "
        f"T{args.start_time_index}->T{args.start_time_index + 1} through "
        f"T{args.end_time_index}->T{args.end_time_index + 1}"
    )
    print("Number of rollout steps:", num_rollout_steps)
    print(
        "Checkpoint best epoch:",
        checkpoint_metadata["checkpoint_best_epoch"],
    )
    print(
        "Checkpoint best valid loss:",
        checkpoint_metadata["checkpoint_best_valid_loss"],
    )
    print("Checkpoint rainfall scale:", checkpoint_rainfall_scale)
    print(
        "Checkpoint time range: "
        f"T{checkpoint_metadata['checkpoint_start_time_index']} through "
        f"T{checkpoint_metadata['checkpoint_end_time_index']}"
    )
    print("Clamp output:", args.clamp_output)
    print("Clamp feedback:", args.clamp_feedback)
    print("Save predictions:", args.save_predictions)

    prepare_output_dir(output_dir, dataset_root, args.overwrite)
    teacher_prediction_dir = None
    autoregressive_prediction_dir = None
    if args.save_predictions:
        teacher_prediction_dir = predictions_root / "teacher_forced"
        autoregressive_prediction_dir = predictions_root / "autoregressive"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    static_device_graph = static_graph_to_device(static_graph, device)
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

    (
        teacher_forced_summary,
        teacher_forced_event_metrics,
        teacher_forced_csi,
    ) = evaluate_teacher_forced(
        model,
        static_device_graph,
        prepared_valid_events,
        time_indices,
        args.rainfall_scale,
        args.clamp_output,
        args.clamp_feedback,
        teacher_prediction_dir,
    )
    write_event_metrics(
        teacher_forced_csv,
        teacher_forced_event_metrics,
    )
    print_mode_diagnostics(
        "Teacher-forced validation",
        teacher_forced_summary,
        teacher_forced_csi,
    )

    (
        autoregressive_summary,
        autoregressive_event_metrics,
        autoregressive_csi,
    ) = evaluate_autoregressive_rollout(
        model,
        static_device_graph,
        prepared_valid_events,
        time_indices,
        args.rainfall_scale,
        args.clamp_output,
        args.clamp_feedback,
        autoregressive_prediction_dir,
    )
    write_event_metrics(
        autoregressive_csv,
        autoregressive_event_metrics,
    )
    print_mode_diagnostics(
        "Autoregressive validation",
        autoregressive_summary,
        autoregressive_csi,
    )

    rollout_degradation = {
        "mean_combined_MSE_ratio": safe_degradation_ratio(
            autoregressive_summary["mean_combined_MSE"],
            teacher_forced_summary["mean_combined_MSE"],
            "mean_combined_MSE",
        ),
        "mean_H_RMSE_ratio": safe_degradation_ratio(
            autoregressive_summary["mean_H_RMSE"],
            teacher_forced_summary["mean_H_RMSE"],
            "mean_H_RMSE",
        ),
        "mean_H_peak_ratio_ratio": safe_degradation_ratio(
            autoregressive_summary["mean_H_peak_ratio"],
            teacher_forced_summary["mean_H_peak_ratio"],
            "mean_H_peak_ratio",
        ),
    }
    print("\nRollout degradation ratios:")
    print(
        "combined MSE ratio:",
        optional_float_text(
            rollout_degradation["mean_combined_MSE_ratio"]
        ),
    )
    print(
        "H RMSE ratio:",
        optional_float_text(rollout_degradation["mean_H_RMSE_ratio"]),
    )
    print(
        "H peak ratio ratio:",
        optional_float_text(
            rollout_degradation["mean_H_peak_ratio_ratio"]
        ),
    )

    cuda_device_name = None
    max_memory_allocated_mb = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)
        max_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (
            1024**2
        )
    total_elapsed_seconds = time.perf_counter() - run_start
    csi_summary = {
        "teacher_forced": teacher_forced_csi,
        "autoregressive": autoregressive_csi,
    }
    summary = {
        "script_name": Path(__file__).name,
        "dataset_root": str(dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "valid_event_ids": valid_event_ids,
        "num_valid_events": len(valid_event_ids),
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "rainfall_scale": args.rainfall_scale,
        "checkpoint_rainfall_scale": checkpoint_rainfall_scale,
        "start_time_index": args.start_time_index,
        "end_time_index": args.end_time_index,
        "num_rollout_steps": num_rollout_steps,
        "checkpoint_best_epoch": checkpoint_metadata[
            "checkpoint_best_epoch"
        ],
        "checkpoint_best_valid_loss": checkpoint_metadata[
            "checkpoint_best_valid_loss"
        ],
        "hid_features": args.hid_features,
        "K": args.K,
        "seed": args.seed,
        "clamp_output": args.clamp_output,
        "clamp_feedback": args.clamp_feedback,
        "save_predictions": args.save_predictions,
        "teacher_forced_summary": teacher_forced_summary,
        "autoregressive_summary": autoregressive_summary,
        "rollout_degradation": rollout_degradation,
        "csi_thresholds": list(CSI_THRESHOLDS.values()),
        "csi_summary": csi_summary,
        "device": str(device),
        "total_elapsed_seconds": total_elapsed_seconds,
    }
    if device.type == "cuda":
        summary["cuda_device_name"] = cuda_device_name
        summary["max_memory_allocated_mb"] = max_memory_allocated_mb
    with summary_json.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)

    print("\nOutput paths:")
    for path in output_paths:
        print(path)
    if args.save_predictions:
        print(teacher_prediction_dir)
        print(autoregressive_prediction_dir)
    if device.type == "cuda":
        print("\nCUDA diagnostics:")
        print("device name:", cuda_device_name)
        print(f"max memory allocated: {max_memory_allocated_mb:.2f} MB")
    print()
    print(
        "Stockbridge SWE-GNN milestone 14B-2 validation rollout "
        "diagnostic passed."
    )


if __name__ == "__main__":
    main()
