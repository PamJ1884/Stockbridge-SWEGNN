import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data


VALID_PKL = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_graph_subset_valid_v01"
    / "stockbridge_valid_graph_subset_003.pkl"
)
CHECKPOINT_PATH = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone11_subset_checkpoint_v01"
    / "milestone11_best_checkpoint.pt"
)
SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone12_rollout_smoke_v01"
)

EXPECTED_VALID_EVENT_IDS = ["R002", "R006", "R011"]
EXPECTED_NUM_NODES = 99430
EXPECTED_X_SHAPE = (99430, 7)
EXPECTED_TARGET_SHAPE = (99430, 2)
EXPECTED_EDGE_INDEX_SHAPE = (2, 396458)
EXPECTED_EDGE_ATTR_SHAPE = (396458, 3)

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

METRIC_FIELDNAMES = [
    "H_MSE",
    "H_RMSE",
    "H_MAE",
    "Qmag_MSE",
    "Qmag_RMSE",
    "Qmag_MAE",
    "combined_MSE",
    "true_H_max",
    "pred_H_max",
    "true_Qmag_max",
    "pred_Qmag_max",
    "raw_pred_H_min",
    "raw_pred_Qmag_min",
    "raw_pred_H_max",
    "raw_pred_Qmag_max",
    "rainfall_raw_t",
    "rainfall_raw_t_plus_1",
    "rainfall_scaled_t",
    "rainfall_scaled_t_plus_1",
]
ROLLOUT_FIELDNAMES = ["t", "target_t", *METRIC_FIELDNAMES]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for milestone 12."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an autoregressive validation-event rollout from the milestone "
            "11 best checkpoint."
        )
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use (default: auto).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the full-graph rollout to run on CPU.",
    )
    parser.add_argument("--event-id", type=str, default="R002")
    parser.add_argument("--start-time-index", type=int, default=2)
    parser.add_argument("--end-time-index", type=int, default=16)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--clamp-nonnegative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp predicted H and Qmag to zero before feedback and metrics.",
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save predicted and true rollout tensors (default: enabled).",
    )
    return parser.parse_args()


def select_device(requested_device: str, allow_cpu: bool) -> torch.device:
    """Resolve the device and protect against accidental CPU rollout."""
    if requested_device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = requested_device

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    if device_type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Refusing to run the full-graph milestone "
            "12 rollout on CPU unless --allow-cpu is set."
        )

    return torch.device(device_type)


def calculate_dem_slopes(
    dem: torch.Tensor,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate x and y DEM slopes using finite differences in grid units."""
    dem_grid = dem.float().reshape(ny, nx)
    slope_x = torch.empty_like(dem_grid)
    slope_y = torch.empty_like(dem_grid)

    slope_x[:, 1:-1] = (dem_grid[:, 2:] - dem_grid[:, :-2]) / 2.0
    slope_x[:, 0] = dem_grid[:, 1] - dem_grid[:, 0]
    slope_x[:, -1] = dem_grid[:, -1] - dem_grid[:, -2]

    slope_y[1:-1, :] = (dem_grid[2:, :] - dem_grid[:-2, :]) / 2.0
    slope_y[0, :] = dem_grid[1, :] - dem_grid[0, :]
    slope_y[-1, :] = dem_grid[-1, :] - dem_grid[-2, :]

    return slope_x.reshape(-1), slope_y.reshape(-1)


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Assert that a tensor contains neither NaN nor infinite values."""
    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN/inf values"


def validate_graph(
    graph: Data,
    expected_event_id: str,
    expected_split: str,
) -> None:
    """Validate one graph from the prepared validation subset."""
    assert isinstance(graph, Data), (
        f"Expected PyG Data for {expected_event_id}, got {type(graph).__name__}"
    )

    required_attributes = (
        "H",
        "Qmag",
        "DEM",
        "edge_index",
        "edge_relative_distance",
        "edge_slope",
        "rainfall_global",
        "pos",
        "ny",
        "nx",
        "nt",
        "num_nodes",
        "event_id",
        "split",
    )
    missing = [name for name in required_attributes if not hasattr(graph, name)]
    assert not missing, f"{expected_event_id} is missing attributes: {missing}"

    expected_shapes = {
        "H": (99430, 25),
        "Qmag": (99430, 25),
        "DEM": (99430,),
        "edge_index": EXPECTED_EDGE_INDEX_SHAPE,
        "edge_relative_distance": (396458, 2),
        "edge_slope": (396458,),
        "rainfall_global": (25,),
        "pos": (99430, 2),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(getattr(graph, name).shape)
        assert actual_shape == expected_shape, (
            f"Expected {expected_event_id} {name} shape {expected_shape}, "
            f"got {actual_shape}"
        )

    assert graph.ny == 305, f"Expected {expected_event_id} ny=305, got {graph.ny}"
    assert graph.nx == 326, f"Expected {expected_event_id} nx=326, got {graph.nx}"
    assert graph.nt == 25, f"Expected {expected_event_id} nt=25, got {graph.nt}"
    assert graph.num_nodes == EXPECTED_NUM_NODES, (
        f"Expected {expected_event_id} num_nodes={EXPECTED_NUM_NODES}, "
        f"got {graph.num_nodes}"
    )
    assert graph.event_id == expected_event_id, (
        f"Expected event_id={expected_event_id}, got {graph.event_id}"
    )
    assert graph.split == expected_split, (
        f"Expected {expected_event_id} split={expected_split}, got {graph.split}"
    )

    for name in (
        "H",
        "Qmag",
        "DEM",
        "edge_relative_distance",
        "edge_slope",
        "rainfall_global",
        "pos",
    ):
        assert_finite(f"{expected_event_id} {name}", getattr(graph, name))


def load_graph_subset(
    path: Path,
    expected_event_ids: list[str],
    expected_split: str,
) -> list[Data]:
    """Load and validate one three-event graph subset."""
    assert path.exists(), f"Graph subset pickle not found: {path}"
    with path.open("rb") as file:
        graphs = pickle.load(file)

    assert isinstance(graphs, list), (
        f"Expected list in {path}, got {type(graphs).__name__}"
    )
    assert len(graphs) == 3, f"Expected 3 graphs in {path}, got {len(graphs)}"

    for graph, expected_event_id in zip(graphs, expected_event_ids):
        validate_graph(graph, expected_event_id, expected_split)

    actual_event_ids = [graph.event_id for graph in graphs]
    assert actual_event_ids == expected_event_ids, (
        f"Expected {expected_split} event order {expected_event_ids}, "
        f"got {actual_event_ids}"
    )
    return graphs


def prepare_events(
    graphs: list[Data],
) -> list[tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Precompute DEM slopes and edge attributes for each event."""
    prepared_events = []
    for graph in graphs:
        slope_x, slope_y = calculate_dem_slopes(
            graph.DEM,
            ny=int(graph.ny),
            nx=int(graph.nx),
        )
        edge_attr = torch.cat(
            [graph.edge_relative_distance, graph.edge_slope.view(-1, 1)],
            dim=1,
        ).float()

        assert tuple(edge_attr.shape) == EXPECTED_EDGE_ATTR_SHAPE, (
            f"Expected {graph.event_id} edge_attr shape "
            f"{EXPECTED_EDGE_ATTR_SHAPE}, got {tuple(edge_attr.shape)}"
        )
        assert_finite(f"{graph.event_id} slope_x", slope_x)
        assert_finite(f"{graph.event_id} slope_y", slope_y)
        assert_finite(f"{graph.event_id} edge_attr", edge_attr)
        prepared_events.append((graph, slope_x, slope_y, edge_attr))

    return prepared_events


def extract_prediction(model_output, context: str) -> torch.Tensor:
    """Extract the first tensor when a model returns a tuple or list."""
    if isinstance(model_output, (tuple, list)):
        assert model_output, f"Model returned an empty tuple/list during {context}"
        model_output = model_output[0]
    assert isinstance(model_output, torch.Tensor), (
        f"Expected prediction tensor during {context}, "
        f"got {type(model_output).__name__}"
    )
    return model_output


def summary_to_dict(
    rows: list[dict[str, float | int]],
    fieldnames: list[str],
) -> dict[str, float]:
    """Return the mean of each requested metric over rollout rows."""
    assert rows, "Cannot summarize an empty rollout"
    return {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in fieldnames
    }


def build_rollout_sample(
    graph: Data,
    slope_x: torch.Tensor,
    slope_y: torch.Tensor,
    edge_attr: torch.Tensor,
    current_H: torch.Tensor,
    current_Qmag: torch.Tensor,
    t: int,
    rainfall_scale: float,
) -> Data:
    """Build one autoregressive rainfall-forced sample for t to t+1."""
    assert current_H.device.type == "cpu", "current_H must remain on CPU"
    assert current_Qmag.device.type == "cpu", "current_Qmag must remain on CPU"
    assert tuple(current_H.shape) == (EXPECTED_NUM_NODES,)
    assert tuple(current_Qmag.shape) == (EXPECTED_NUM_NODES,)

    target = torch.stack(
        [graph.H[:, t + 1], graph.Qmag[:, t + 1]],
        dim=1,
    )
    rainfall_t_scaled = graph.rainfall_global[t] * rainfall_scale
    rainfall_next_scaled = graph.rainfall_global[t + 1] * rainfall_scale
    rainfall_t_node = torch.full_like(graph.DEM, rainfall_t_scaled.item())
    rainfall_next_node = torch.full_like(
        graph.DEM,
        rainfall_next_scaled.item(),
    )

    x = torch.stack(
        [
            slope_x,
            slope_y,
            rainfall_t_node,
            rainfall_next_node,
            graph.DEM,
            current_H,
            current_Qmag,
        ],
        dim=1,
    )

    sample = Data()
    sample.edge_index = graph.edge_index
    sample.edge_attr = edge_attr.float()
    sample.x = x.float()
    sample.y = target.float()
    sample.pos = graph.pos.float()
    sample.DEM = graph.DEM.float()
    sample.input_time_index = t
    sample.target_time_index = t + 1
    sample.previous_t = 1
    sample.event_id = graph.event_id
    sample.split = graph.split
    sample.ny = graph.ny
    sample.nx = graph.nx
    sample.nt = graph.nt
    sample.num_nodes = graph.num_nodes
    sample.rainfall_input_raw = graph.rainfall_global[t].float()
    sample.rainfall_target_raw = graph.rainfall_global[t + 1].float()
    sample.rainfall_input_scaled = rainfall_t_scaled.float()
    sample.rainfall_target_scaled = rainfall_next_scaled.float()
    sample.rainfall_scale = float(rainfall_scale)
    sample.input_features = INPUT_FEATURES
    sample.edge_features = EDGE_FEATURES
    sample.output_variables = OUTPUT_VARIABLES

    assert tuple(sample.x.shape) == EXPECTED_X_SHAPE, (
        f"Expected x shape {EXPECTED_X_SHAPE}, got {tuple(sample.x.shape)}"
    )
    assert tuple(sample.y.shape) == EXPECTED_TARGET_SHAPE, (
        f"Expected y shape {EXPECTED_TARGET_SHAPE}, got {tuple(sample.y.shape)}"
    )
    assert tuple(sample.edge_index.shape) == EXPECTED_EDGE_INDEX_SHAPE, (
        "Expected edge_index shape "
        f"{EXPECTED_EDGE_INDEX_SHAPE}, got {tuple(sample.edge_index.shape)}"
    )
    assert tuple(sample.edge_attr.shape) == EXPECTED_EDGE_ATTR_SHAPE, (
        "Expected edge_attr shape "
        f"{EXPECTED_EDGE_ATTR_SHAPE}, got {tuple(sample.edge_attr.shape)}"
    )
    assert_finite("rollout x", sample.x)
    assert_finite("rollout y", sample.y)
    assert_finite("rollout edge_attr", sample.edge_attr)
    return sample


def scalar(value):
    """Convert scalar checkpoint metadata to a JSON-safe Python value."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected scalar checkpoint metadata tensor")
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def checkpoint_model_config(checkpoint: dict) -> dict:
    """Recover and validate the milestone 11 model configuration."""
    saved = checkpoint.get("model_config", {})
    if saved is None:
        saved = {}
    if not isinstance(saved, dict):
        raise TypeError("checkpoint model_config must be a dictionary")

    node_features = int(saved.get("node_features", checkpoint.get("node_features", 7)))
    edge_features = int(
        saved.get("edge_features", checkpoint.get("edge_feature_count", 3))
    )
    fixed_expected = {
        "node_features": 7,
        "edge_features": 3,
        "type_GNN": "SWEGNN",
        "gnn_activation": "tanh",
        "dropout": 0.0,
        "mlp_layers": 2,
        "mlp_activation": "prelu",
        "with_filter_matrix": True,
        "with_gradient": True,
        "with_WL": True,
        "previous_t": 1,
    }
    recovered = {
        "node_features": node_features,
        "edge_features": edge_features,
        "type_GNN": saved.get("type_GNN", "SWEGNN"),
        "gnn_activation": saved.get("gnn_activation", "tanh"),
        "dropout": float(saved.get("dropout", 0.0)),
        "mlp_layers": int(saved.get("mlp_layers", 2)),
        "mlp_activation": saved.get("mlp_activation", "prelu"),
        "with_filter_matrix": bool(saved.get("with_filter_matrix", True)),
        "with_gradient": bool(saved.get("with_gradient", True)),
        "with_WL": bool(saved.get("with_WL", True)),
        "previous_t": int(saved.get("previous_t", checkpoint.get("previous_t", 1))),
    }
    mismatches = {
        name: (recovered[name], expected)
        for name, expected in fixed_expected.items()
        if recovered[name] != expected
    }
    if mismatches:
        raise ValueError(
            "Checkpoint model configuration does not match milestone 12: "
            f"{mismatches}"
        )

    recovered["hid_features"] = int(saved.get("hid_features", 16))
    recovered["K"] = int(saved.get("K", 4))
    recovered["seed"] = int(saved.get("seed", 4444))
    if recovered["hid_features"] <= 0:
        raise ValueError("Checkpoint hid_features must be greater than zero")
    if recovered["K"] <= 0:
        raise ValueError("Checkpoint K must be greater than zero")
    return recovered


def calculate_step_metrics(
    raw_pred: torch.Tensor,
    pred_H: torch.Tensor,
    pred_Qmag: torch.Tensor,
    target: torch.Tensor,
    graph: Data,
    t: int,
    rainfall_scale: float,
) -> dict[str, float | int]:
    """Calculate requested rollout diagnostics for one target timestep."""
    true_H = target[:, 0]
    true_Qmag = target[:, 1]
    H_error = pred_H - true_H
    Qmag_error = pred_Qmag - true_Qmag
    H_MSE = torch.mean(H_error.square())
    Qmag_MSE = torch.mean(Qmag_error.square())
    combined_MSE = torch.mean(
        torch.stack([H_error, Qmag_error], dim=1).square()
    )

    row = {
        "t": t,
        "target_t": t + 1,
        "H_MSE": H_MSE.item(),
        "H_RMSE": torch.sqrt(H_MSE).item(),
        "H_MAE": torch.mean(torch.abs(H_error)).item(),
        "Qmag_MSE": Qmag_MSE.item(),
        "Qmag_RMSE": torch.sqrt(Qmag_MSE).item(),
        "Qmag_MAE": torch.mean(torch.abs(Qmag_error)).item(),
        "combined_MSE": combined_MSE.item(),
        "true_H_max": true_H.max().item(),
        "pred_H_max": pred_H.max().item(),
        "true_Qmag_max": true_Qmag.max().item(),
        "pred_Qmag_max": pred_Qmag.max().item(),
        "raw_pred_H_min": raw_pred[:, 0].min().item(),
        "raw_pred_Qmag_min": raw_pred[:, 1].min().item(),
        "raw_pred_H_max": raw_pred[:, 0].max().item(),
        "raw_pred_Qmag_max": raw_pred[:, 1].max().item(),
        "rainfall_raw_t": graph.rainfall_global[t].item(),
        "rainfall_raw_t_plus_1": graph.rainfall_global[t + 1].item(),
        "rainfall_scaled_t": graph.rainfall_global[t].item() * rainfall_scale,
        "rainfall_scaled_t_plus_1": (
            graph.rainfall_global[t + 1].item() * rainfall_scale
        ),
    }
    assert all(
        math.isfinite(float(row[name])) for name in METRIC_FIELDNAMES
    ), f"Non-finite metric at input timestep {t}"
    return row


def print_step_metrics(row: dict[str, float | int]) -> None:
    """Print one concise rollout-step diagnostic line."""
    print(
        f"t={row['t']} target_t={row['target_t']} "
        f"H_RMSE={row['H_RMSE']:.8g} H_MAE={row['H_MAE']:.8g} "
        f"Qmag_RMSE={row['Qmag_RMSE']:.8g} "
        f"Qmag_MAE={row['Qmag_MAE']:.8g} "
        f"true_H_max={row['true_H_max']:.8g} "
        f"pred_H_max={row['pred_H_max']:.8g} "
        f"rainfall_scaled_t={row['rainfall_scaled_t']:.8g} "
        f"rainfall_scaled_t_plus_1={row['rainfall_scaled_t_plus_1']:.8g}"
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device, args.allow_cpu)

    if not math.isfinite(args.rainfall_scale):
        raise ValueError("--rainfall-scale must be finite")
    if not SWEGNN_REPO.is_dir():
        raise FileNotFoundError(f"Original SWE-GNN repo not found: {SWEGNN_REPO}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)

    from models.gnn import GNN

    valid_graphs = load_graph_subset(
        VALID_PKL,
        EXPECTED_VALID_EVENT_IDS,
        "valid",
    )
    matching_graphs = [
        graph for graph in valid_graphs if graph.event_id == args.event_id
    ]
    if not matching_graphs:
        raise ValueError(
            f"Unknown --event-id {args.event_id!r}; expected one of "
            f"{EXPECTED_VALID_EVENT_IDS}"
        )
    graph = matching_graphs[0]

    if args.start_time_index < 0:
        raise ValueError("--start-time-index must be greater than or equal to zero")
    if args.end_time_index >= int(graph.nt) - 1:
        raise ValueError(
            f"--end-time-index must be less than {int(graph.nt) - 1}; "
            f"got {args.end_time_index}"
        )
    if args.start_time_index > args.end_time_index:
        raise ValueError("--start-time-index must not exceed --end-time-index")

    metrics_csv = (
        args.output_dir
        / f"milestone12_rollout_metrics_{args.event_id}.csv"
    )
    summary_json = (
        args.output_dir
        / f"milestone12_rollout_summary_{args.event_id}.json"
    )
    predictions_pt = (
        args.output_dir
        / f"milestone12_rollout_predictions_{args.event_id}.pt"
    )
    output_paths = [metrics_csv, summary_json]
    if args.save_predictions:
        output_paths.append(predictions_pt)
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.overwrite:
        formatted_paths = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing milestone 12 outputs without "
            f"--overwrite:\n{formatted_paths}"
        )

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dictionary, got {type(checkpoint).__name__}"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict")
    model_config = checkpoint_model_config(checkpoint)

    model = GNN(
        node_features=model_config["node_features"],
        edge_features=model_config["edge_features"],
        type_GNN=model_config["type_GNN"],
        hid_features=model_config["hid_features"],
        K=model_config["K"],
        gnn_activation=model_config["gnn_activation"],
        dropout=model_config["dropout"],
        mlp_layers=model_config["mlp_layers"],
        mlp_activation=model_config["mlp_activation"],
        seed=model_config["seed"],
        with_filter_matrix=model_config["with_filter_matrix"],
        with_gradient=model_config["with_gradient"],
        with_WL=model_config["with_WL"],
        previous_t=model_config["previous_t"],
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prepared_event = prepare_events([graph])[0]
    graph, slope_x, slope_y, edge_attr = prepared_event
    pred_time_indices = list(
        range(args.start_time_index + 1, args.end_time_index + 2)
    )
    checkpoint_epoch = scalar(checkpoint.get("epoch"))
    checkpoint_best_epoch = scalar(checkpoint.get("best_epoch"))
    checkpoint_best_valid_loss = scalar(checkpoint.get("best_valid_loss"))

    print("=== Stockbridge SWE-GNN milestone 12 rollout smoke ===")
    print("Selected event:", graph.event_id)
    print("Checkpoint epoch:", checkpoint_epoch)
    print("Best epoch:", checkpoint_best_epoch)
    print("Best valid loss:", checkpoint_best_valid_loss)
    print(
        "Rollout range: "
        f"T{args.start_time_index}->T{args.start_time_index + 1} through "
        f"T{args.end_time_index}->T{args.end_time_index + 1}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    current_H = graph.H[:, args.start_time_index].detach().cpu()
    current_Qmag = graph.Qmag[:, args.start_time_index].detach().cpu()
    predicted_H = []
    predicted_Qmag = []
    rollout_metrics = []

    with torch.inference_mode():
        for t in range(args.start_time_index, args.end_time_index + 1):
            sample = build_rollout_sample(
                graph,
                slope_x,
                slope_y,
                edge_attr,
                current_H,
                current_Qmag,
                t,
                args.rainfall_scale,
            ).to(device)
            raw_pred = extract_prediction(
                model(sample),
                f"rollout for {graph.event_id} at T{t}",
            )
            assert tuple(raw_pred.shape) == EXPECTED_TARGET_SHAPE, (
                f"Expected prediction shape {EXPECTED_TARGET_SHAPE}, "
                f"got {tuple(raw_pred.shape)}"
            )
            assert_finite(f"{graph.event_id} rollout prediction at T{t}", raw_pred)

            if args.clamp_nonnegative:
                pred_H = raw_pred[:, 0].clamp_min(0)
                pred_Qmag = raw_pred[:, 1].clamp_min(0)
            else:
                pred_H = raw_pred[:, 0]
                pred_Qmag = raw_pred[:, 1]

            row = calculate_step_metrics(
                raw_pred,
                pred_H,
                pred_Qmag,
                sample.y,
                graph,
                t,
                args.rainfall_scale,
            )
            rollout_metrics.append(row)
            print_step_metrics(row)

            current_H = pred_H.detach().cpu()
            current_Qmag = pred_Qmag.detach().cpu()
            predicted_H.append(current_H)
            predicted_Qmag.append(current_Qmag)

    pred_H_tensor = torch.stack(predicted_H, dim=0)
    pred_Qmag_tensor = torch.stack(predicted_Qmag, dim=0)
    true_H_tensor = torch.stack(
        [graph.H[:, t].detach().cpu() for t in pred_time_indices],
        dim=0,
    )
    true_Qmag_tensor = torch.stack(
        [graph.Qmag[:, t].detach().cpu() for t in pred_time_indices],
        dim=0,
    )
    expected_rollout_shape = (len(pred_time_indices), EXPECTED_NUM_NODES)
    for name, tensor in (
        ("pred_H", pred_H_tensor),
        ("pred_Qmag", pred_Qmag_tensor),
        ("true_H", true_H_tensor),
        ("true_Qmag", true_Qmag_tensor),
    ):
        assert tuple(tensor.shape) == expected_rollout_shape, (
            f"Expected {name} shape {expected_rollout_shape}, "
            f"got {tuple(tensor.shape)}"
        )
        assert_finite(name, tensor)

    mean_metrics = summary_to_dict(rollout_metrics, METRIC_FIELDNAMES)
    cuda_device_name = None
    max_memory_allocated_mb = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)
        max_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (
            1024**2
        )

    summary = {
        "script_name": Path(__file__).name,
        "event_id": graph.event_id,
        "valid_pkl": str(VALID_PKL),
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_epoch": checkpoint_best_epoch,
        "checkpoint_best_valid_loss": checkpoint_best_valid_loss,
        "output_dir": str(args.output_dir),
        "input_features": INPUT_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "rainfall_scale": args.rainfall_scale,
        "start_time_index": args.start_time_index,
        "end_time_index": args.end_time_index,
        "pred_time_indices": pred_time_indices,
        "clamp_nonnegative": args.clamp_nonnegative,
        "final_step_metrics": rollout_metrics[-1],
        "mean_metrics_over_rollout": mean_metrics,
        "max_pred_H_over_rollout": max(
            float(row["pred_H_max"]) for row in rollout_metrics
        ),
        "max_true_H_over_rollout": max(
            float(row["true_H_max"]) for row in rollout_metrics
        ),
        "max_pred_Qmag_over_rollout": max(
            float(row["pred_Qmag_max"]) for row in rollout_metrics
        ),
        "max_true_Qmag_over_rollout": max(
            float(row["true_Qmag_max"]) for row in rollout_metrics
        ),
        "device": str(device),
        "cuda_device_name": cuda_device_name,
        "max_memory_allocated_mb": max_memory_allocated_mb,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=ROLLOUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rollout_metrics)

    with summary_json.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)

    if args.save_predictions:
        predictions = {
            "event_id": graph.event_id,
            "start_time_index": args.start_time_index,
            "end_time_index": args.end_time_index,
            "pred_time_indices": pred_time_indices,
            "pred_H": pred_H_tensor,
            "pred_Qmag": pred_Qmag_tensor,
            "true_H": true_H_tensor,
            "true_Qmag": true_Qmag_tensor,
            "rainfall_global": graph.rainfall_global.detach().cpu(),
            "input_features": INPUT_FEATURES,
            "output_variables": OUTPUT_VARIABLES,
            "clamp_nonnegative": args.clamp_nonnegative,
        }
        torch.save(predictions, predictions_pt)

    print("Saved metrics:", metrics_csv)
    print("Saved summary:", summary_json)
    if args.save_predictions:
        print("Saved predictions:", predictions_pt)
    print("Stockbridge SWE-GNN milestone 12 rollout smoke passed.")


if __name__ == "__main__":
    main()
