import argparse
import csv
import json
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
SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-SWEGNN"
    / "milestone11_subset_checkpoint_v01"
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

HISTORY_FIELDNAMES = [
    "epoch",
    "train_update_mean",
    "train_update_min",
    "train_update_max",
    "train_eval_mean",
    "train_eval_min",
    "train_eval_max",
    "valid_eval_mean",
    "valid_eval_min",
    "valid_eval_max",
    "epoch_elapsed_seconds",
    "best_valid_so_far",
    "is_best",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for milestone 11."""
    parser = argparse.ArgumentParser(
        description="Train SWE-GNN on small train/valid subsets with checkpoints."
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
        help="Allow the full-graph checkpoint experiment to run on CPU.",
    )
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--start-time-index", type=int, default=2)
    parser.add_argument("--end-time-index", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    shuffle_group = parser.add_mutually_exclusive_group()
    shuffle_group.add_argument("--shuffle", dest="shuffle", action="store_true")
    shuffle_group.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
    )
    parser.set_defaults(shuffle=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
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
            "CUDA is not available. Refusing to run full-graph milestone 11 "
            "training on CPU unless --allow-cpu is set."
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
    """Validate one graph from a prepared subset."""
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


def build_one_step_sample(
    graph: Data,
    slope_x: torch.Tensor,
    slope_y: torch.Tensor,
    edge_attr: torch.Tensor,
    t: int,
    rainfall_scale: float,
) -> Data:
    """Build an in-memory rainfall-forced sample for timestep t to t+1."""
    H_t = graph.H[:, t]
    Qmag_t = graph.Qmag[:, t]
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
            H_t,
            Qmag_t,
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
    sample.rainfall_global = graph.rainfall_global.float()
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
    assert_finite("x", sample.x)
    assert_finite("y", sample.y)
    assert_finite("edge_attr", sample.edge_attr)

    return sample


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
            f"Expected {graph.event_id} edge_attr shape {EXPECTED_EDGE_ATTR_SHAPE}, "
            f"got {tuple(edge_attr.shape)}"
        )
        assert_finite(f"{graph.event_id} slope_x", slope_x)
        assert_finite(f"{graph.event_id} slope_y", slope_y)
        assert_finite(f"{graph.event_id} edge_attr", edge_attr)
        prepared_events.append((graph, slope_x, slope_y, edge_attr))

    return prepared_events


def extract_prediction(model_output, context: str):
    """Extract the first tensor when a model returns a tuple or list."""
    if isinstance(model_output, (tuple, list)):
        assert model_output, f"Model returned an empty tuple/list during {context}"
        return model_output[0]
    return model_output


def summarize_losses(losses: list[float]) -> tuple[float, float, float]:
    """Return mean, minimum, and maximum losses."""
    assert losses, "Cannot summarize an empty loss list"
    return sum(losses) / len(losses), min(losses), max(losses)


def summary_to_dict(
    summary: tuple[float, float, float],
) -> dict[str, float]:
    """Convert a loss summary tuple to a named dictionary."""
    mean_loss, min_loss, max_loss = summary
    return {
        "mean": mean_loss,
        "min": min_loss,
        "max": max_loss,
    }


def evaluate_model(
    model: torch.nn.Module,
    prepared_events: list[
        tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    event_time_pairs: list[tuple[int, int]],
    rainfall_scale: float,
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[float, float, float]:
    """Evaluate all selected event/timestep pairs without model updates."""
    losses = []
    model.eval()

    with torch.no_grad():
        for event_index, t in event_time_pairs:
            graph, slope_x, slope_y, edge_attr = prepared_events[event_index]
            sample = build_one_step_sample(
                graph,
                slope_x,
                slope_y,
                edge_attr,
                t,
                rainfall_scale,
            )
            sample = sample.to(device)
            pred = extract_prediction(
                model(sample),
                f"evaluation for {graph.event_id} at T{t}",
            )

            assert isinstance(pred, torch.Tensor), (
                f"Expected prediction tensor, got {type(pred).__name__}"
            )
            assert tuple(pred.shape) == EXPECTED_TARGET_SHAPE, (
                "Expected prediction shape "
                f"{EXPECTED_TARGET_SHAPE}, got {tuple(pred.shape)}"
            )
            assert_finite(f"{graph.event_id} evaluation prediction at T{t}", pred)

            loss = criterion(pred, sample.y)
            assert torch.isfinite(loss).item(), (
                f"Non-finite evaluation loss for {graph.event_id} at T{t}"
            )
            losses.append(loss.item())

    return summarize_losses(losses)


def build_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_eval_summary: tuple[float, float, float],
    valid_eval_summary: tuple[float, float, float],
    best_valid_loss: float,
    best_epoch: int,
    args: argparse.Namespace,
    train_event_ids: list[str],
    valid_event_ids: list[str],
    node_features: int,
    edge_feature_count: int,
    previous_t: int,
    model_config: dict,
) -> dict:
    """Build a self-describing checkpoint dictionary."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_eval_summary": summary_to_dict(train_eval_summary),
        "valid_eval_summary": summary_to_dict(valid_eval_summary),
        "best_valid_loss": best_valid_loss,
        "best_epoch": best_epoch,
        "args": vars(args).copy(),
        "train_event_ids": train_event_ids,
        "valid_event_ids": valid_event_ids,
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "node_features": node_features,
        "edge_feature_count": edge_feature_count,
        "previous_t": previous_t,
        "rainfall_scale": args.rainfall_scale,
        "start_time_index": args.start_time_index,
        "end_time_index": args.end_time_index,
        "model_config": model_config,
    }


def print_loss_summary(
    label: str,
    summary: tuple[float, float, float],
) -> None:
    """Print mean, minimum, and maximum loss with a shared format."""
    mean_loss, min_loss, max_loss = summary
    print(
        f"{label}: mean={mean_loss:.8g} "
        f"min={min_loss:.8g} max={max_loss:.8g}"
    )


def main() -> None:
    args = parse_args()

    history_csv = args.output_dir / "milestone11_train_valid_history.csv"
    best_checkpoint = args.output_dir / "milestone11_best_checkpoint.pt"
    final_checkpoint = args.output_dir / "milestone11_final_checkpoint.pt"
    metadata_json = args.output_dir / "milestone11_run_metadata.json"
    output_paths = [
        history_csv,
        best_checkpoint,
        final_checkpoint,
        metadata_json,
    ]

    print("=== Stockbridge SWE-GNN milestone 11 checkpoint training smoke ===")
    print("Train PKL:", TRAIN_PKL)
    print("Valid PKL:", VALID_PKL)
    print("Original SWE-GNN repo:", SWEGNN_REPO)
    print("Output directory:", args.output_dir)

    device = select_device(args.device, args.allow_cpu)
    print("Selected device:", device)

    assert args.hid_features > 0, "--hid-features must be greater than zero"
    assert args.K > 0, "--K must be greater than zero"
    assert args.epochs > 0, "--epochs must be greater than zero"
    assert args.lr > 0.0, "--lr must be greater than zero"
    assert args.weight_decay >= 0.0, "--weight-decay must not be negative"
    assert torch.isfinite(torch.tensor(args.rainfall_scale)).item(), (
        "--rainfall-scale must be finite"
    )
    assert SWEGNN_REPO.is_dir(), f"Original SWE-GNN repo not found: {SWEGNN_REPO}"

    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)

    from models.gnn import GNN

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

    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.overwrite:
        formatted_paths = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing milestone 11 outputs without "
            f"--overwrite:\n{formatted_paths}"
        )

    if args.start_time_index < 0:
        raise ValueError("--start-time-index must be greater than or equal to zero")
    if args.end_time_index >= int(train_graphs[0].nt) - 1:
        raise ValueError(
            f"--end-time-index must be less than {int(train_graphs[0].nt) - 1}; "
            f"got {args.end_time_index}"
        )
    if args.start_time_index > args.end_time_index:
        raise ValueError("--start-time-index must not exceed --end-time-index")

    time_indices = list(range(args.start_time_index, args.end_time_index + 1))
    train_pairs = [
        (event_index, t)
        for event_index in range(len(train_graphs))
        for t in time_indices
    ]
    valid_pairs = [
        (event_index, t)
        for event_index in range(len(valid_graphs))
        for t in time_indices
    ]

    prepared_train_events = prepare_events(train_graphs)
    prepared_valid_events = prepare_events(valid_graphs)
    train_event_ids = [graph.event_id for graph in train_graphs]
    valid_event_ids = [graph.event_id for graph in valid_graphs]

    print("Train event IDs:", train_event_ids)
    print("Valid event IDs:", valid_event_ids)
    print("Selected timesteps:", time_indices)
    print("Train pairs:", len(train_pairs))
    print("Valid pairs:", len(valid_pairs))
    print("Input features:", INPUT_FEATURES)
    print("Rainfall scale:", args.rainfall_scale)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    node_features = 7
    edge_feature_count = 3
    previous_t = 1
    model_config = {
        "node_features": node_features,
        "edge_features": edge_feature_count,
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
        "previous_t": previous_t,
        "device": str(device),
    }
    model = GNN(
        node_features=node_features,
        edge_features=edge_feature_count,
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
        previous_t=previous_t,
        device=device,
    ).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.MSELoss()

    first_parameter_name, first_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    first_parameter_before = first_parameter.detach().clone()

    print("\nModel configuration:")
    print(model_config)
    print("parameter count:", parameter_count)
    print("optimizer: Adam")
    print("learning rate:", args.lr)
    print("weight decay:", args.weight_decay)
    print("shuffle:", args.shuffle)

    initial_train = evaluate_model(
        model,
        prepared_train_events,
        train_pairs,
        args.rainfall_scale,
        device,
        criterion,
    )
    initial_valid = evaluate_model(
        model,
        prepared_valid_events,
        valid_pairs,
        args.rainfall_scale,
        device,
        criterion,
    )
    print("\nInitial evaluation:")
    print_loss_summary("Initial train", initial_train)
    print_loss_summary("Initial valid", initial_valid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_valid_loss = float("inf")
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
            epoch_train_pairs = train_pairs.copy()
            if args.shuffle:
                random.shuffle(epoch_train_pairs)

            epoch_losses = []
            model.train()

            for event_index, t in epoch_train_pairs:
                graph, slope_x, slope_y, edge_attr = (
                    prepared_train_events[event_index]
                )
                sample = build_one_step_sample(
                    graph,
                    slope_x,
                    slope_y,
                    edge_attr,
                    t,
                    args.rainfall_scale,
                )
                sample = sample.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = extract_prediction(
                    model(sample),
                    f"training for {graph.event_id} at T{t}",
                )

                assert isinstance(pred, torch.Tensor), (
                    f"Expected prediction tensor, got {type(pred).__name__}"
                )
                assert tuple(pred.shape) == EXPECTED_TARGET_SHAPE, (
                    "Expected prediction shape "
                    f"{EXPECTED_TARGET_SHAPE}, got {tuple(pred.shape)}"
                )
                assert_finite(
                    f"{graph.event_id} training prediction at T{t}",
                    pred,
                )

                loss = criterion(pred, sample.y)
                assert torch.isfinite(loss).item(), (
                    f"Non-finite training loss for {graph.event_id} at T{t}"
                )
                loss.backward()

                parameters_with_gradients = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                assert parameters_with_gradients, (
                    "No model parameter received a gradient for "
                    f"{graph.event_id} at T{t}"
                )
                gradients_are_finite = all(
                    torch.isfinite(parameter.grad).all().item()
                    for parameter in parameters_with_gradients
                )
                assert gradients_are_finite, (
                    f"Non-finite gradient for {graph.event_id} at T{t}"
                )

                optimizer.step()
                epoch_losses.append(loss.item())

            train_update_summary = summarize_losses(epoch_losses)
            train_eval_summary = evaluate_model(
                model,
                prepared_train_events,
                train_pairs,
                args.rainfall_scale,
                device,
                criterion,
            )
            valid_eval_summary = evaluate_model(
                model,
                prepared_valid_events,
                valid_pairs,
                args.rainfall_scale,
                device,
                criterion,
            )
            epoch_elapsed_seconds = time.perf_counter() - epoch_start

            is_best = valid_eval_summary[0] < best_valid_loss
            if is_best:
                best_valid_loss = valid_eval_summary[0]
                best_epoch = epoch
                checkpoint = build_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    train_eval_summary=train_eval_summary,
                    valid_eval_summary=valid_eval_summary,
                    best_valid_loss=best_valid_loss,
                    best_epoch=best_epoch,
                    args=args,
                    train_event_ids=train_event_ids,
                    valid_event_ids=valid_event_ids,
                    node_features=node_features,
                    edge_feature_count=edge_feature_count,
                    previous_t=previous_t,
                    model_config=model_config,
                )
                torch.save(checkpoint, best_checkpoint)

            history_writer.writerow(
                {
                    "epoch": epoch,
                    "train_update_mean": train_update_summary[0],
                    "train_update_min": train_update_summary[1],
                    "train_update_max": train_update_summary[2],
                    "train_eval_mean": train_eval_summary[0],
                    "train_eval_min": train_eval_summary[1],
                    "train_eval_max": train_eval_summary[2],
                    "valid_eval_mean": valid_eval_summary[0],
                    "valid_eval_min": valid_eval_summary[1],
                    "valid_eval_max": valid_eval_summary[2],
                    "epoch_elapsed_seconds": epoch_elapsed_seconds,
                    "best_valid_so_far": best_valid_loss,
                    "is_best": is_best,
                }
            )
            history_file.flush()

            print(f"\nEpoch {epoch}/{args.epochs}")
            print_loss_summary("Train update", train_update_summary)
            print_loss_summary("Train evaluation", train_eval_summary)
            print_loss_summary("Valid evaluation", valid_eval_summary)
            print("best checkpoint saved:", is_best)
            print(f"elapsed seconds: {epoch_elapsed_seconds:.2f}")

    final_train_eval = evaluate_model(
        model,
        prepared_train_events,
        train_pairs,
        args.rainfall_scale,
        device,
        criterion,
    )
    final_valid_eval = evaluate_model(
        model,
        prepared_valid_events,
        valid_pairs,
        args.rainfall_scale,
        device,
        criterion,
    )

    assert best_epoch > 0, "No best checkpoint was saved"
    first_parameter_absolute_change = torch.sum(
        torch.abs(first_parameter.detach() - first_parameter_before)
    ).item()
    assert first_parameter_absolute_change > 0.0, (
        f"First trainable parameter did not change: {first_parameter_name}"
    )

    final_checkpoint_data = build_checkpoint(
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        train_eval_summary=final_train_eval,
        valid_eval_summary=final_valid_eval,
        best_valid_loss=best_valid_loss,
        best_epoch=best_epoch,
        args=args,
        train_event_ids=train_event_ids,
        valid_event_ids=valid_event_ids,
        node_features=node_features,
        edge_feature_count=edge_feature_count,
        previous_t=previous_t,
        model_config=model_config,
    )
    torch.save(final_checkpoint_data, final_checkpoint)

    cuda_device_name = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)

    run_metadata = {
        "script_name": Path(__file__).name,
        "train_pkl": str(TRAIN_PKL),
        "valid_pkl": str(VALID_PKL),
        "output_dir": str(args.output_dir),
        "train_event_ids": train_event_ids,
        "valid_event_ids": valid_event_ids,
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
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "final_train_eval": summary_to_dict(final_train_eval),
        "final_valid_eval": summary_to_dict(final_valid_eval),
        "device": str(device),
        "cuda_device_name": cuda_device_name,
    }
    with metadata_json.open("w") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2)

    print("\nFinal evaluation:")
    print_loss_summary("Final train", final_train_eval)
    print_loss_summary("Final valid", final_valid_eval)
    print("best epoch:", best_epoch)
    print(f"best valid loss: {best_valid_loss:.8g}")
    print("first trainable parameter:", first_parameter_name)
    print(
        "first parameter absolute change: "
        f"{first_parameter_absolute_change:.8g}"
    )

    print("\nSaved outputs:")
    for path in output_paths:
        print(path)

    if device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print("\nCUDA diagnostics:")
        print("device name:", cuda_device_name)
        print(f"max memory allocated: {max_memory_mb:.2f} MB")

    print()
    print("Stockbridge SWE-GNN milestone 11 checkpoint training smoke passed.")


if __name__ == "__main__":
    main()
