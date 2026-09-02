import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data


TRAIN_PKL = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_graph_subset_train_v01"
    / "stockbridge_train_graph_subset_003.pkl"
)
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
    / "milestone12d_autoregressive_finetune_v01"
)

EXPECTED_TRAIN_EVENT_IDS = ["R001", "R003", "R004"]
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
EVALUATION_MODES = ("teacher_forced", "autoregressive")

HISTORY_FIELDNAMES = [
    "epoch",
    "train_multistep_loss_mean",
    "train_multistep_loss_min",
    "train_multistep_loss_max",
    "valid_autoreg_mean_H_RMSE",
    "valid_autoreg_mean_H_MAE",
    "valid_autoreg_mean_Qmag_RMSE",
    "valid_autoreg_mean_Qmag_MAE",
    "valid_autoreg_mean_combined_MSE",
    "valid_autoreg_mean_H_peak_ratio",
    "valid_autoreg_mean_Qmag_peak_ratio",
    "best_valid_autoreg_combined_MSE_so_far",
    "is_best",
    "epoch_elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for milestone 12D."""
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the milestone 11 checkpoint with differentiable short "
            "autoregressive rollouts on the Stockbridge subsets."
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
        help="Allow the full-graph fine-tuning smoke test to run on CPU.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-time-index", type=int, default=2)
    parser.add_argument("--end-time-index", type=int, default=16)
    parser.add_argument("--unroll-steps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clamp-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp H and Qmag to zero before autoregressive feedback/metrics.",
    )
    parser.add_argument(
        "--loss-on-clamped",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute training loss on nonnegative-clamped predictions.",
    )
    return parser.parse_args()


def select_device(requested_device: str, allow_cpu: bool) -> torch.device:
    """Resolve the device and protect against accidental CPU fine-tuning."""
    if requested_device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = requested_device

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    if device_type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Refusing to run full-graph milestone 12D "
            "fine-tuning on CPU unless --allow-cpu is set."
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
    """Validate one graph from a prepared Stockbridge subset."""
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


def prepare_events(graphs: list[Data], device: torch.device) -> list[dict]:
    """Precompute and copy all repeatedly used event tensors to the device."""
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

        prepared = {
            "event_id": graph.event_id,
            "split": graph.split,
            "ny": int(graph.ny),
            "nx": int(graph.nx),
            "nt": int(graph.nt),
            "num_nodes": int(graph.num_nodes),
            "H": graph.H.float().to(device),
            "Qmag": graph.Qmag.float().to(device),
            "DEM": graph.DEM.float().to(device),
            "slope_x": slope_x.float().to(device),
            "slope_y": slope_y.float().to(device),
            "edge_index": graph.edge_index.long().to(device),
            "edge_attr": edge_attr.to(device),
            "rainfall_global": graph.rainfall_global.float().to(device),
            "pos": graph.pos.float().to(device),
        }
        for tensor_name in (
            "H",
            "Qmag",
            "DEM",
            "slope_x",
            "slope_y",
            "edge_attr",
            "rainfall_global",
            "pos",
        ):
            assert_finite(
                f"{graph.event_id} prepared {tensor_name}",
                prepared[tensor_name],
            )
        prepared_events.append(prepared)
    return prepared_events


def build_device_sample(
    event: dict,
    current_H: torch.Tensor,
    current_Qmag: torch.Tensor,
    t: int,
    rainfall_scale: float,
) -> Data:
    """Build a PyG sample directly from tensors already on the target device."""
    device = event["DEM"].device
    assert current_H.device == device, "current_H is not on the event device"
    assert current_Qmag.device == device, "current_Qmag is not on the event device"
    assert tuple(current_H.shape) == (EXPECTED_NUM_NODES,)
    assert tuple(current_Qmag.shape) == (EXPECTED_NUM_NODES,)

    rainfall_t_scaled = event["rainfall_global"][t] * rainfall_scale
    rainfall_next_scaled = event["rainfall_global"][t + 1] * rainfall_scale
    x = torch.stack(
        [
            event["slope_x"],
            event["slope_y"],
            rainfall_t_scaled.expand_as(event["DEM"]),
            rainfall_next_scaled.expand_as(event["DEM"]),
            event["DEM"],
            current_H,
            current_Qmag,
        ],
        dim=1,
    )
    target = torch.stack(
        [event["H"][:, t + 1], event["Qmag"][:, t + 1]],
        dim=1,
    )

    sample = Data()
    sample.edge_index = event["edge_index"]
    sample.edge_attr = event["edge_attr"]
    sample.x = x
    sample.y = target
    sample.pos = event["pos"]
    sample.DEM = event["DEM"]
    sample.input_time_index = t
    sample.target_time_index = t + 1
    sample.previous_t = 1
    sample.event_id = event["event_id"]
    sample.split = event["split"]
    sample.ny = event["ny"]
    sample.nx = event["nx"]
    sample.nt = event["nt"]
    sample.num_nodes = event["num_nodes"]
    sample.rainfall_input_raw = event["rainfall_global"][t]
    sample.rainfall_target_raw = event["rainfall_global"][t + 1]
    sample.rainfall_input_scaled = rainfall_t_scaled
    sample.rainfall_target_scaled = rainfall_next_scaled
    sample.rainfall_scale = float(rainfall_scale)
    sample.rainfall_global = event["rainfall_global"]
    sample.input_features = INPUT_FEATURES
    sample.edge_features = EDGE_FEATURES
    sample.output_variables = OUTPUT_VARIABLES

    assert tuple(sample.x.shape) == EXPECTED_X_SHAPE
    assert tuple(sample.y.shape) == EXPECTED_TARGET_SHAPE
    assert tuple(sample.edge_index.shape) == EXPECTED_EDGE_INDEX_SHAPE
    assert tuple(sample.edge_attr.shape) == EXPECTED_EDGE_ATTR_SHAPE
    assert sample.x.device == device
    assert sample.y.device == device
    assert_finite("sample x", sample.x)
    assert_finite("sample y", sample.y)
    return sample


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
            "Checkpoint model configuration does not match milestone 12D: "
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
    rainfall_global: torch.Tensor,
    t: int,
    rainfall_scale: float,
) -> dict[str, float | int]:
    """Calculate rollout diagnostics for one target timestep."""
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
        "rainfall_raw_t": rainfall_global[t].item(),
        "rainfall_raw_t_plus_1": rainfall_global[t + 1].item(),
        "rainfall_scaled_t": rainfall_global[t].item() * rainfall_scale,
        "rainfall_scaled_t_plus_1": (
            rainfall_global[t + 1].item() * rainfall_scale
        ),
    }
    assert all(math.isfinite(float(value)) for value in row.values()), (
        f"Non-finite metric at input timestep {t}"
    )
    return row


def safe_peak_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite peak ratio, or None when the true peak is zero."""
    if denominator == 0.0:
        return None
    ratio = numerator / denominator
    assert math.isfinite(ratio), "Non-finite predicted/true peak ratio"
    return ratio


def summarize_event_mode(
    event_id: str,
    mode: str,
    rows: list[dict[str, float | int]],
) -> dict[str, object]:
    """Summarize per-step metrics for one event and evaluation mode."""
    assert rows, f"Cannot summarize empty metrics for {event_id} {mode}"
    final_row = rows[-1]
    max_true_H = max(float(row["true_H_max"]) for row in rows)
    max_pred_H = max(float(row["pred_H_max"]) for row in rows)
    max_true_Qmag = max(float(row["true_Qmag_max"]) for row in rows)
    max_pred_Qmag = max(float(row["pred_Qmag_max"]) for row in rows)
    return {
        "event_id": event_id,
        "mode": mode,
        "mean_H_RMSE": sum(float(row["H_RMSE"]) for row in rows) / len(rows),
        "mean_H_MAE": sum(float(row["H_MAE"]) for row in rows) / len(rows),
        "mean_Qmag_RMSE": (
            sum(float(row["Qmag_RMSE"]) for row in rows) / len(rows)
        ),
        "mean_Qmag_MAE": (
            sum(float(row["Qmag_MAE"]) for row in rows) / len(rows)
        ),
        "mean_combined_MSE": (
            sum(float(row["combined_MSE"]) for row in rows) / len(rows)
        ),
        "final_H_RMSE": float(final_row["H_RMSE"]),
        "final_H_MAE": float(final_row["H_MAE"]),
        "final_Qmag_RMSE": float(final_row["Qmag_RMSE"]),
        "final_Qmag_MAE": float(final_row["Qmag_MAE"]),
        "max_true_H_over_rollout": max_true_H,
        "max_pred_H_over_rollout": max_pred_H,
        "max_true_Qmag_over_rollout": max_true_Qmag,
        "max_pred_Qmag_over_rollout": max_pred_Qmag,
        "pred_true_H_peak_ratio": safe_peak_ratio(max_pred_H, max_true_H),
        "pred_true_Qmag_peak_ratio": safe_peak_ratio(
            max_pred_Qmag,
            max_true_Qmag,
        ),
    }


def mean_optional(values: list[float | None]) -> float | None:
    """Average defined values while retaining None when all are undefined."""
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def summarize_evaluation(
    event_summaries: list[dict[str, object]],
) -> dict[str, object]:
    """Average event summaries and retain the underlying per-event rows."""
    assert event_summaries, "Cannot summarize an empty evaluation"
    return {
        "mean_H_RMSE": sum(
            float(row["mean_H_RMSE"]) for row in event_summaries
        )
        / len(event_summaries),
        "mean_H_MAE": sum(
            float(row["mean_H_MAE"]) for row in event_summaries
        )
        / len(event_summaries),
        "mean_Qmag_RMSE": sum(
            float(row["mean_Qmag_RMSE"]) for row in event_summaries
        )
        / len(event_summaries),
        "mean_Qmag_MAE": sum(
            float(row["mean_Qmag_MAE"]) for row in event_summaries
        )
        / len(event_summaries),
        "mean_combined_MSE": sum(
            float(row["mean_combined_MSE"]) for row in event_summaries
        )
        / len(event_summaries),
        "mean_H_peak_ratio": mean_optional(
            [row["pred_true_H_peak_ratio"] for row in event_summaries]
        ),
        "mean_Qmag_peak_ratio": mean_optional(
            [row["pred_true_Qmag_peak_ratio"] for row in event_summaries]
        ),
        "event_summaries": event_summaries,
    }


def evaluate_events(
    model: torch.nn.Module,
    prepared_events: list[dict],
    mode: str,
    start_time_index: int,
    end_time_index: int,
    rainfall_scale: float,
    clamp_feedback: bool,
) -> dict[str, object]:
    """Evaluate all events in teacher-forced or autoregressive mode."""
    if mode not in EVALUATION_MODES:
        raise ValueError(
            f"Unknown evaluation mode {mode!r}; expected {EVALUATION_MODES}"
        )

    model.eval()
    event_summaries = []
    with torch.inference_mode():
        for event in prepared_events:
            current_H = event["H"][:, start_time_index]
            current_Qmag = event["Qmag"][:, start_time_index]
            rows = []
            for t in range(start_time_index, end_time_index + 1):
                if mode == "teacher_forced":
                    current_H = event["H"][:, t]
                    current_Qmag = event["Qmag"][:, t]

                sample = build_device_sample(
                    event,
                    current_H,
                    current_Qmag,
                    t,
                    rainfall_scale,
                )
                raw_pred = extract_prediction(
                    model(sample),
                    f"{mode} evaluation for {event['event_id']} at T{t}",
                )
                assert tuple(raw_pred.shape) == EXPECTED_TARGET_SHAPE, (
                    f"Expected prediction shape {EXPECTED_TARGET_SHAPE}, "
                    f"got {tuple(raw_pred.shape)}"
                )
                assert_finite(
                    f"{event['event_id']} {mode} prediction at T{t}",
                    raw_pred,
                )

                if clamp_feedback:
                    pred_H = raw_pred[:, 0].clamp_min(0)
                    pred_Qmag = raw_pred[:, 1].clamp_min(0)
                else:
                    pred_H = raw_pred[:, 0]
                    pred_Qmag = raw_pred[:, 1]
                rows.append(
                    calculate_step_metrics(
                        raw_pred,
                        pred_H,
                        pred_Qmag,
                        sample.y,
                        event["rainfall_global"],
                        t,
                        rainfall_scale,
                    )
                )
                if mode == "autoregressive":
                    current_H = pred_H
                    current_Qmag = pred_Qmag

            event_summaries.append(
                summarize_event_mode(event["event_id"], mode, rows)
            )
    return summarize_evaluation(event_summaries)


def summarize_losses(losses: list[float]) -> tuple[float, float, float]:
    """Return mean, minimum, and maximum losses."""
    assert losses, "Cannot summarize an empty loss list"
    return sum(losses) / len(losses), min(losses), max(losses)


def train_multistep_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    prepared_events: list[dict],
    training_windows: list[tuple[int, int]],
    unroll_steps: int,
    rainfall_scale: float,
    clamp_feedback: bool,
    loss_on_clamped: bool,
    shuffle: bool,
) -> tuple[float, float, float]:
    """Train one epoch, backpropagating once through each complete unroll."""
    epoch_windows = training_windows.copy()
    if shuffle:
        random.shuffle(epoch_windows)

    model.train()
    window_losses = []
    for event_index, start_t in epoch_windows:
        event = prepared_events[event_index]
        current_H = event["H"][:, start_t]
        current_Qmag = event["Qmag"][:, start_t]
        step_losses = []

        optimizer.zero_grad(set_to_none=True)
        for k in range(unroll_steps):
            t = start_t + k
            sample = build_device_sample(
                event,
                current_H,
                current_Qmag,
                t,
                rainfall_scale,
            )
            raw_pred = extract_prediction(
                model(sample),
                f"training for {event['event_id']} window T{start_t} step T{t}",
            )
            assert tuple(raw_pred.shape) == EXPECTED_TARGET_SHAPE, (
                f"Expected prediction shape {EXPECTED_TARGET_SHAPE}, "
                f"got {tuple(raw_pred.shape)}"
            )
            assert_finite(
                f"{event['event_id']} training prediction at T{t}",
                raw_pred,
            )

            if loss_on_clamped:
                loss_pred_H = raw_pred[:, 0].clamp_min(0)
                loss_pred_Qmag = raw_pred[:, 1].clamp_min(0)
            else:
                loss_pred_H = raw_pred[:, 0]
                loss_pred_Qmag = raw_pred[:, 1]
            loss_prediction = torch.stack(
                [loss_pred_H, loss_pred_Qmag],
                dim=1,
            )
            step_loss = criterion(loss_prediction, sample.y)
            assert torch.isfinite(step_loss).item(), (
                f"Non-finite training loss for {event['event_id']} at T{t}"
            )
            step_losses.append(step_loss)

            if clamp_feedback:
                current_H = raw_pred[:, 0].clamp_min(0)
                current_Qmag = raw_pred[:, 1].clamp_min(0)
            else:
                current_H = raw_pred[:, 0]
                current_Qmag = raw_pred[:, 1]

        window_loss = torch.stack(step_losses).mean()
        window_loss.backward()
        parameters_with_gradients = [
            parameter
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert parameters_with_gradients, (
            "No model parameter received a gradient for "
            f"{event['event_id']} window starting at T{start_t}"
        )
        assert all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in parameters_with_gradients
        ), (
            "Non-finite gradient for "
            f"{event['event_id']} window starting at T{start_t}"
        )
        optimizer.step()
        window_losses.append(window_loss.item())

    return summarize_losses(window_losses)


def build_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_valid_autoreg_combined_MSE: float,
    args: argparse.Namespace,
    train_event_ids: list[str],
    valid_event_ids: list[str],
    model_config: dict,
    valid_autoreg_summary: dict[str, object],
) -> dict:
    """Build a self-describing milestone 12D checkpoint dictionary."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_valid_autoreg_combined_MSE": best_valid_autoreg_combined_MSE,
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
        "unroll_steps": args.unroll_steps,
        "clamp_feedback": args.clamp_feedback,
        "loss_on_clamped": args.loss_on_clamped,
        "valid_autoreg_summary": valid_autoreg_summary,
    }


def print_event_summaries(summary: dict[str, object]) -> None:
    """Print the requested concise per-event rollout diagnostics."""
    for row in summary["event_summaries"]:
        ratio = row["pred_true_H_peak_ratio"]
        ratio_text = "undefined" if ratio is None else f"{float(ratio):.8g}"
        print(
            f"event_id={row['event_id']} "
            f"mean_H_RMSE={float(row['mean_H_RMSE']):.8g} "
            f"final_H_RMSE={float(row['final_H_RMSE']):.8g} "
            f"max_true_H={float(row['max_true_H_over_rollout']):.8g} "
            f"max_pred_H={float(row['max_pred_H_over_rollout']):.8g} "
            f"H_peak_ratio={ratio_text}"
        )


def optional_metric_text(value: object) -> str:
    """Format an optional floating-point aggregate for diagnostics."""
    if value is None:
        return "undefined"
    return f"{float(value):.8g}"


def validate_time_arguments(
    args: argparse.Namespace,
    train_graphs: list[Data],
    valid_graphs: list[Data],
) -> list[int]:
    """Validate the evaluation range and derive valid training starts."""
    if args.start_time_index < 0:
        raise ValueError("--start-time-index must be greater than or equal to zero")
    if args.start_time_index > args.end_time_index:
        raise ValueError("--start-time-index must not exceed --end-time-index")
    if args.unroll_steps <= 0:
        raise ValueError("--unroll-steps must be greater than zero")
    for graph in [*train_graphs, *valid_graphs]:
        if args.end_time_index >= int(graph.nt) - 1:
            raise ValueError(
                f"--end-time-index must be less than {int(graph.nt) - 1} for "
                f"{graph.event_id}; got {args.end_time_index}"
            )

    largest_start_t = args.end_time_index + 1 - args.unroll_steps
    if largest_start_t < args.start_time_index:
        raise ValueError(
            "--unroll-steps does not fit in the requested training time range: "
            f"T{args.start_time_index} through T{args.end_time_index}"
        )
    training_start_indices = list(
        range(args.start_time_index, largest_start_t + 1)
    )
    assert training_start_indices[-1] + args.unroll_steps <= (
        args.end_time_index + 1
    )
    return training_start_indices


def main() -> None:
    args = parse_args()
    device = select_device(args.device, args.allow_cpu)

    history_csv = args.output_dir / "milestone12d_history.csv"
    best_checkpoint_path = args.output_dir / "milestone12d_best_checkpoint.pt"
    final_checkpoint_path = args.output_dir / "milestone12d_final_checkpoint.pt"
    summary_json = args.output_dir / "milestone12d_summary.json"
    output_paths = [
        history_csv,
        best_checkpoint_path,
        final_checkpoint_path,
        summary_json,
    ]

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("--lr must be finite and greater than zero")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be finite and nonnegative")
    if not math.isfinite(args.rainfall_scale):
        raise ValueError("--rainfall-scale must be finite")
    if not SWEGNN_REPO.is_dir():
        raise FileNotFoundError(f"Original SWE-GNN repo not found: {SWEGNN_REPO}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.overwrite:
        formatted_paths = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing milestone 12D outputs without "
            f"--overwrite:\n{formatted_paths}"
        )

    assert INPUT_FEATURES[-3:] == ["DEM", "H_t", "Qmag_t"], (
        "with_WL=True requires DEM immediately before the final H_t/Qmag_t columns"
    )
    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)
    from models.gnn import GNN

    print("=== Stockbridge SWE-GNN milestone 12D autoregressive fine-tuning ===")
    print("Train PKL:", TRAIN_PKL)
    print("Valid PKL:", VALID_PKL)
    print("Checkpoint:", args.checkpoint_path)
    print("Output directory:", args.output_dir)
    print("Selected device:", device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_graphs = load_graph_subset(
        TRAIN_PKL,
        EXPECTED_TRAIN_EVENT_IDS,
        "train",
    )
    valid_graphs = load_graph_subset(
        VALID_PKL,
        EXPECTED_VALID_EVENT_IDS,
        "valid",
    )
    training_start_indices = validate_time_arguments(
        args,
        train_graphs,
        valid_graphs,
    )
    training_windows = [
        (event_index, start_t)
        for event_index in range(len(train_graphs))
        for start_t in training_start_indices
    ]
    train_event_ids = [graph.event_id for graph in train_graphs]
    valid_event_ids = [graph.event_id for graph in valid_graphs]

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dictionary, got {type(checkpoint).__name__}"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict")
    model_config = checkpoint_model_config(checkpoint)
    checkpoint_epoch = scalar(checkpoint.get("epoch"))
    checkpoint_best_epoch = scalar(checkpoint.get("best_epoch"))
    checkpoint_best_valid_loss = scalar(checkpoint.get("best_valid_loss"))

    torch.manual_seed(model_config["seed"])
    random.seed(model_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_config["seed"])

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.MSELoss()
    prepared_train_events = prepare_events(train_graphs, device)
    prepared_valid_events = prepare_events(valid_graphs, device)

    first_parameter_name, first_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    first_parameter_before = first_parameter.detach().clone()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("Train event IDs:", train_event_ids)
    print("Valid event IDs:", valid_event_ids)
    print(
        "Training window starts: "
        f"T{training_start_indices[0]} to T{training_start_indices[-1]} inclusive"
    )
    print("Training windows per epoch:", len(training_windows))
    print(
        "Validation rollout: "
        f"T{args.start_time_index}->T{args.start_time_index + 1} through "
        f"T{args.end_time_index}->T{args.end_time_index + 1}"
    )
    print("Input features:", INPUT_FEATURES)
    print("Model configuration:", model_config)
    print("Parameter count:", parameter_count)
    print("Checkpoint epoch:", checkpoint_epoch)
    print("Checkpoint best epoch:", checkpoint_best_epoch)
    print("Checkpoint best valid loss:", checkpoint_best_valid_loss)
    print("Unroll steps:", args.unroll_steps)
    print("Clamp feedback:", args.clamp_feedback)
    print("Loss on clamped predictions:", args.loss_on_clamped)

    initial_valid_autoreg_summary = evaluate_events(
        model,
        prepared_valid_events,
        "autoregressive",
        args.start_time_index,
        args.end_time_index,
        args.rainfall_scale,
        args.clamp_feedback,
    )
    print("\nInitial valid autoregressive summary per event:")
    print_event_summaries(initial_valid_autoreg_summary)
    print(
        "Initial valid autoregressive aggregate: "
        f"mean_H_RMSE={float(initial_valid_autoreg_summary['mean_H_RMSE']):.8g} "
        "mean_combined_MSE="
        f"{float(initial_valid_autoreg_summary['mean_combined_MSE']):.8g} "
        "mean_H_peak_ratio="
        f"{optional_metric_text(initial_valid_autoreg_summary['mean_H_peak_ratio'])} "
        "mean_Qmag_peak_ratio="
        f"{optional_metric_text(initial_valid_autoreg_summary['mean_Qmag_peak_ratio'])}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_valid_autoreg_combined_MSE = float("inf")
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
            train_loss_summary = train_multistep_epoch(
                model,
                optimizer,
                criterion,
                prepared_train_events,
                training_windows,
                args.unroll_steps,
                args.rainfall_scale,
                args.clamp_feedback,
                args.loss_on_clamped,
                args.shuffle,
            )
            valid_autoreg_summary = evaluate_events(
                model,
                prepared_valid_events,
                "autoregressive",
                args.start_time_index,
                args.end_time_index,
                args.rainfall_scale,
                args.clamp_feedback,
            )
            valid_autoregressive_mean_combined_MSE = float(
                valid_autoreg_summary["mean_combined_MSE"]
            )
            is_best = (
                valid_autoregressive_mean_combined_MSE
                < best_valid_autoreg_combined_MSE
            )
            if is_best:
                best_valid_autoreg_combined_MSE = (
                    valid_autoregressive_mean_combined_MSE
                )
                best_epoch = epoch
                torch.save(
                    build_checkpoint(
                        epoch,
                        model,
                        optimizer,
                        best_valid_autoreg_combined_MSE,
                        args,
                        train_event_ids,
                        valid_event_ids,
                        model_config,
                        valid_autoreg_summary,
                    ),
                    best_checkpoint_path,
                )

            epoch_elapsed_seconds = time.perf_counter() - epoch_start
            history_writer.writerow(
                {
                    "epoch": epoch,
                    "train_multistep_loss_mean": train_loss_summary[0],
                    "train_multistep_loss_min": train_loss_summary[1],
                    "train_multistep_loss_max": train_loss_summary[2],
                    "valid_autoreg_mean_H_RMSE": valid_autoreg_summary[
                        "mean_H_RMSE"
                    ],
                    "valid_autoreg_mean_H_MAE": valid_autoreg_summary[
                        "mean_H_MAE"
                    ],
                    "valid_autoreg_mean_Qmag_RMSE": valid_autoreg_summary[
                        "mean_Qmag_RMSE"
                    ],
                    "valid_autoreg_mean_Qmag_MAE": valid_autoreg_summary[
                        "mean_Qmag_MAE"
                    ],
                    "valid_autoreg_mean_combined_MSE": (
                        valid_autoregressive_mean_combined_MSE
                    ),
                    "valid_autoreg_mean_H_peak_ratio": valid_autoreg_summary[
                        "mean_H_peak_ratio"
                    ],
                    "valid_autoreg_mean_Qmag_peak_ratio": valid_autoreg_summary[
                        "mean_Qmag_peak_ratio"
                    ],
                    "best_valid_autoreg_combined_MSE_so_far": (
                        best_valid_autoreg_combined_MSE
                    ),
                    "is_best": is_best,
                    "epoch_elapsed_seconds": epoch_elapsed_seconds,
                }
            )
            history_file.flush()

            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                "train multistep loss: "
                f"mean={train_loss_summary[0]:.8g} "
                f"min={train_loss_summary[1]:.8g} "
                f"max={train_loss_summary[2]:.8g}"
            )
            print(
                "valid autoregressive: "
                f"mean_H_RMSE={float(valid_autoreg_summary['mean_H_RMSE']):.8g} "
                "mean_combined_MSE="
                f"{valid_autoregressive_mean_combined_MSE:.8g} "
                "mean_H_peak_ratio="
                f"{optional_metric_text(valid_autoreg_summary['mean_H_peak_ratio'])} "
                "mean_Qmag_peak_ratio="
                f"{optional_metric_text(valid_autoreg_summary['mean_Qmag_peak_ratio'])}"
            )
            print("best checkpoint saved:", is_best)
            print(f"elapsed seconds: {epoch_elapsed_seconds:.2f}")

    final_valid_autoreg_summary = evaluate_events(
        model,
        prepared_valid_events,
        "autoregressive",
        args.start_time_index,
        args.end_time_index,
        args.rainfall_scale,
        args.clamp_feedback,
    )
    final_valid_teacher_forced_summary = evaluate_events(
        model,
        prepared_valid_events,
        "teacher_forced",
        args.start_time_index,
        args.end_time_index,
        args.rainfall_scale,
        args.clamp_feedback,
    )
    assert best_epoch > 0, "No best checkpoint was saved"
    first_parameter_absolute_change = torch.sum(
        torch.abs(first_parameter.detach() - first_parameter_before)
    ).item()
    assert first_parameter_absolute_change > 0.0, (
        f"First trainable parameter did not change: {first_parameter_name}"
    )

    torch.save(
        build_checkpoint(
            args.epochs,
            model,
            optimizer,
            best_valid_autoreg_combined_MSE,
            args,
            train_event_ids,
            valid_event_ids,
            model_config,
            final_valid_autoreg_summary,
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
        "train_pkl": str(TRAIN_PKL),
        "valid_pkl": str(VALID_PKL),
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_epoch": checkpoint_best_epoch,
        "checkpoint_best_valid_loss": checkpoint_best_valid_loss,
        "output_dir": str(args.output_dir),
        "train_event_ids": train_event_ids,
        "valid_event_ids": valid_event_ids,
        "input_features": INPUT_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "rainfall_scale": args.rainfall_scale,
        "start_time_index": args.start_time_index,
        "end_time_index": args.end_time_index,
        "unroll_steps": args.unroll_steps,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "clamp_feedback": args.clamp_feedback,
        "loss_on_clamped": args.loss_on_clamped,
        "best_epoch": best_epoch,
        "best_valid_autoreg_combined_MSE": best_valid_autoreg_combined_MSE,
        "initial_valid_autoreg_summary": initial_valid_autoreg_summary,
        "final_valid_autoreg_summary": final_valid_autoreg_summary,
        "final_valid_teacher_forced_summary": (
            final_valid_teacher_forced_summary
        ),
        "device": str(device),
    }
    if device.type == "cuda":
        summary["cuda_device_name"] = cuda_device_name
        summary["max_memory_allocated_mb"] = max_memory_allocated_mb
    with summary_json.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)

    print("\nFinal valid autoregressive summary per event:")
    print_event_summaries(final_valid_autoreg_summary)
    print("\nFinal valid teacher-forced summary per event:")
    print_event_summaries(final_valid_teacher_forced_summary)
    print("Best epoch:", best_epoch)
    print(
        "Best valid autoregressive combined MSE: "
        f"{best_valid_autoreg_combined_MSE:.8g}"
    )
    print("First trainable parameter:", first_parameter_name)
    print(
        "First parameter absolute change: "
        f"{first_parameter_absolute_change:.8g}"
    )
    print("\nSaved outputs:")
    for path in output_paths:
        print(path)
    if device.type == "cuda":
        print("\nCUDA diagnostics:")
        print("device name:", cuda_device_name)
        print(f"max memory allocated: {max_memory_allocated_mb:.2f} MB")
    print()
    print("Stockbridge SWE-GNN milestone 12D autoregressive fine-tuning smoke passed.")


if __name__ == "__main__":
    main()
