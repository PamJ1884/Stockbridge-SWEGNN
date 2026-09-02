import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data


DATASET_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_swegnn_full_v01"
)
SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"

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
MODEL_SEED = 4444


def parse_args() -> argparse.Namespace:
    """Parse command-line options for milestone 13B."""
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the shared-static full Stockbridge SWE-GNN dataset "
            "with train, validation, and test samples."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow full-graph SWE-GNN forward passes to run on CPU.",
    )
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument("--time-index", type=int, default=6)
    parser.add_argument("--train-event-id", type=str, default="R001")
    parser.add_argument("--valid-event-id", type=str, default="R002")
    parser.add_argument("--test-event-id", type=str, default="R019")
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    return parser.parse_args()


def select_device(requested_device: str, allow_cpu: bool) -> torch.device:
    """Resolve the device and protect against accidental CPU forward passes."""
    if requested_device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = requested_device

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    if device_type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Refusing to run full-graph milestone 13B "
            "forward passes on CPU unless --allow-cpu is set."
        )
    return torch.device(device_type)


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
            f"Expected static graph cell_size to be numeric, got "
            f"{type(cell_size).__name__}"
        )
    if not math.isfinite(float(cell_size)) or float(cell_size) <= 0.0:
        raise ValueError(
            f"Expected static graph cell_size to be finite and positive, "
            f"got {cell_size}"
        )

    for name, expected_shape in EXPECTED_STATIC_SHAPES.items():
        tensor = static_graph[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected static graph {name} to be a tensor, "
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
        raise TypeError(
            f"Expected integer static graph edge_index, got {edge_index.dtype}"
        )
    row, col = edge_index
    if (row == col).any().item():
        raise ValueError("Static graph edge_index contains self-loops")
    return static_graph


def load_manifest(dataset_root: Path) -> dict:
    """Load and validate the Milestone 13A JSON and CSV manifests."""
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
            f"Expected manifest.json to contain an object, "
            f"got {type(manifest).__name__}"
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
    expected_total_events = sum(EXPECTED_SPLIT_COUNTS.values())
    for key, expected_value in {
        "total_events": expected_total_events,
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
        rows = list(csv.DictReader(manifest_csv_file))
    if len(rows) != expected_total_events:
        raise ValueError(
            f"Expected {expected_total_events} rows in manifest.csv, "
            f"found {len(rows)}"
        )
    return manifest


def load_dynamic_event(
    dataset_root: Path,
    split: str,
    event_id: str,
) -> dict:
    """Load and validate one split-specific dynamic event dictionary."""
    if split not in EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Unknown split {split!r}; expected one of "
            f"{list(EXPECTED_SPLIT_COUNTS)}"
        )
    event_path = dataset_root / "events" / split / f"{event_id}.pt"
    if not event_path.is_file():
        raise FileNotFoundError(
            f"Dynamic event file not found for {split}/{event_id}: {event_path}"
        )
    dynamic_event = torch.load(event_path, map_location="cpu")
    if not isinstance(dynamic_event, dict):
        raise TypeError(
            f"Expected {event_path} to contain a dictionary, got "
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
                f"Expected {split}/{event_id} {name} to be a tensor, "
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
        raise ValueError(
            f"{split}/{event_id} H minimum {H_min:.8g} is below -1e-8"
        )
    if Qmag_min < -1e-8:
        raise ValueError(
            f"{split}/{event_id} Qmag minimum {Qmag_min:.8g} is below -1e-8"
        )
    return dynamic_event


def static_graph_to_device(static_graph: dict, device: torch.device) -> dict:
    """Copy the shared static tensors to the selected device exactly once."""
    device_graph = static_graph.copy()
    for name in ("DEM", "slope_x", "slope_y", "pos", "edge_attr"):
        device_graph[name] = static_graph[name].float().to(device)
    device_graph["edge_index"] = static_graph["edge_index"].long().to(device)
    return device_graph


def build_sample(
    static_graph: dict,
    dynamic_event: dict,
    t: int,
    rainfall_scale: float,
    device: torch.device,
) -> Data:
    """Build one rainfall-forced PyG sample from shared and dynamic tensors."""
    if t < 0 or t >= NT - 1:
        raise ValueError(f"time index must satisfy 0 <= t < {NT - 1}; got {t}")
    if not math.isfinite(rainfall_scale):
        raise ValueError("rainfall_scale must be finite")

    DEM = static_graph["DEM"].float().to(device)
    slope_x = static_graph["slope_x"].float().to(device)
    slope_y = static_graph["slope_y"].float().to(device)
    edge_index = static_graph["edge_index"].long().to(device)
    edge_attr = static_graph["edge_attr"].float().to(device)
    pos = static_graph["pos"].float().to(device)
    H = dynamic_event["H"].float().to(device)
    Qmag = dynamic_event["Qmag"].float().to(device)
    rainfall_global = dynamic_event["rainfall_global"].float().to(device)

    rainfall_t_scaled = rainfall_global[t] * rainfall_scale
    rainfall_next_scaled = rainfall_global[t + 1] * rainfall_scale
    H_t = H[:, t]
    Qmag_t = Qmag[:, t]
    x = torch.stack(
        [
            slope_x,
            slope_y,
            rainfall_t_scaled.expand(NUM_NODES),
            rainfall_next_scaled.expand(NUM_NODES),
            DEM,
            H_t,
            Qmag_t,
        ],
        dim=1,
    )
    y = torch.stack([H[:, t + 1], Qmag[:, t + 1]], dim=1)

    sample = Data()
    sample.x = x
    sample.y = y
    sample.edge_index = edge_index
    sample.edge_attr = edge_attr
    sample.pos = pos
    sample.DEM = DEM
    sample.event_id = dynamic_event["event_id"]
    sample.split = dynamic_event["split"]
    sample.input_time_index = t
    sample.target_time_index = t + 1
    sample.rainfall_input_scaled = rainfall_t_scaled
    sample.rainfall_target_scaled = rainfall_next_scaled
    sample.rainfall_global = rainfall_global
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
        assert actual_shape == expected_shape, (
            f"Expected sample {name} shape {expected_shape}, got {actual_shape}"
        )
    for name in ("x", "y", "edge_attr"):
        assert_finite(
            f"sample {dynamic_event['event_id']} {name}",
            getattr(sample, name),
        )
    assert sample.x.device == device
    assert sample.y.device == device
    assert sample.edge_index.device == device
    assert torch.equal(sample.x[:, -3], DEM), "Feature -3 must be DEM"
    assert torch.equal(sample.x[:, -2], H_t), "Feature -2 must be H_t"
    assert torch.equal(sample.x[:, -1], Qmag_t), "Feature -1 must be Qmag_t"
    assert sample.input_features[-3:] == ["DEM", "H_t", "Qmag_t"]
    return sample


def print_sample_diagnostics(sample: Data) -> None:
    """Print concise shapes and forcing/state diagnostics for one sample."""
    t = int(sample.input_time_index)
    print(f"\nSample {sample.split}/{sample.event_id}")
    print(f"timestep: T{t} -> T{t + 1}")
    print("x shape:", tuple(sample.x.shape))
    print("y shape:", tuple(sample.y.shape))
    print("edge_index shape:", tuple(sample.edge_index.shape))
    print("edge_attr shape:", tuple(sample.edge_attr.shape))
    print(f"rainfall_scaled_t: {sample.rainfall_input_scaled.item():.8g}")
    print(
        "rainfall_scaled_t_plus_1: "
        f"{sample.rainfall_target_scaled.item():.8g}"
    )
    print(f"H_t max: {sample.x[:, 5].max().item():.8g}")
    print(f"target H_t+1 max: {sample.y[:, 0].max().item():.8g}")
    print(f"Qmag_t max: {sample.x[:, 6].max().item():.8g}")
    print(f"target Qmag_t+1 max: {sample.y[:, 1].max().item():.8g}")


def extract_prediction(model_output, context: str) -> torch.Tensor:
    """Extract and validate a tensor from direct or tuple/list model output."""
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


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()
    device = select_device(args.device, args.allow_cpu)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not SWEGNN_REPO.is_dir():
        raise FileNotFoundError(f"Original SWE-GNN repo not found: {SWEGNN_REPO}")
    if not math.isfinite(args.rainfall_scale):
        raise ValueError("--rainfall-scale must be finite")
    if args.time_index < 0 or args.time_index >= NT - 1:
        raise ValueError(
            f"--time-index must satisfy 0 <= index < {NT - 1}; "
            f"got {args.time_index}"
        )
    if args.hid_features <= 0:
        raise ValueError("--hid-features must be greater than zero")
    if args.K <= 0:
        raise ValueError("--K must be greater than zero")

    print("=== Stockbridge SWE-GNN milestone 13B full dataset loader smoke ===")
    print("Dataset root:", dataset_root)
    print("Original SWE-GNN repo:", SWEGNN_REPO)
    print("Selected device:", device)

    static_graph = load_static_graph(dataset_root)
    manifest = load_manifest(dataset_root)
    print("Manifest total events:", manifest["total_events"])
    print("Manifest split counts:", manifest["split_counts"])
    print("R019 present in test:", manifest["contains_R019_in_test"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    static_graph_device = static_graph_to_device(static_graph, device)
    selections = (
        ("train", args.train_event_id),
        ("valid", args.valid_event_id),
        ("test", args.test_event_id),
    )
    samples = []
    for split, event_id in selections:
        dynamic_event = load_dynamic_event(dataset_root, split, event_id)
        sample = build_sample(
            static_graph_device,
            dynamic_event,
            args.time_index,
            args.rainfall_scale,
            device,
        )
        samples.append(sample)
        print_sample_diagnostics(sample)

    first_sample = samples[0]
    for sample in samples[1:]:
        assert sample.edge_index.data_ptr() == first_sample.edge_index.data_ptr(), (
            "Samples do not share the device edge_index tensor"
        )
        assert sample.edge_attr.data_ptr() == first_sample.edge_attr.data_ptr(), (
            "Samples do not share the device edge_attr tensor"
        )
        assert sample.pos.data_ptr() == first_sample.pos.data_ptr(), (
            "Samples do not share the device pos tensor"
        )
        assert sample.DEM.data_ptr() == first_sample.DEM.data_ptr(), (
            "Samples do not share the device DEM tensor"
        )
    print("\nShared static device tensors reused across all three samples: True")

    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)
    from models.gnn import GNN

    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
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
        seed=MODEL_SEED,
        with_filter_matrix=True,
        with_gradient=True,
        with_WL=True,
        previous_t=1,
        device=device,
    ).to(device)
    model.eval()

    print("\nModel configuration:")
    print("node_features: 7")
    print("edge_features: 3")
    print("type_GNN: SWEGNN")
    print("hid_features:", args.hid_features)
    print("K:", args.K)
    print("with_WL: True")
    print("previous_t: 1")
    print(
        "This model is randomly initialized; metrics only test loader/model "
        "compatibility."
    )

    with torch.inference_mode():
        for sample in samples:
            context = f"forward pass for {sample.split}/{sample.event_id}"
            pred = extract_prediction(model(sample), context)
            expected_prediction_shape = (NUM_NODES, 2)
            if tuple(pred.shape) != expected_prediction_shape:
                raise ValueError(
                    f"Expected prediction shape {expected_prediction_shape} "
                    f"during {context}, got {tuple(pred.shape)}"
                )
            assert_finite(f"prediction for {sample.event_id}", pred)
            mse = torch.mean((pred - sample.y).square()).item()
            mae = torch.mean(torch.abs(pred - sample.y)).item()
            if not math.isfinite(mse) or not math.isfinite(mae):
                raise ValueError(f"Non-finite error metric during {context}")

            print(f"\nForward diagnostics {sample.split}/{sample.event_id}")
            print(
                f"pred H min/max: {pred[:, 0].min().item():.8g} / "
                f"{pred[:, 0].max().item():.8g}"
            )
            print(
                f"pred Qmag min/max: {pred[:, 1].min().item():.8g} / "
                f"{pred[:, 1].max().item():.8g}"
            )
            print(f"MSE vs y: {mse:.8g}")
            print(f"MAE vs y: {mae:.8g}")

    print()
    print("Stockbridge SWE-GNN milestone 13B full dataset loader smoke passed.")


if __name__ == "__main__":
    main()
