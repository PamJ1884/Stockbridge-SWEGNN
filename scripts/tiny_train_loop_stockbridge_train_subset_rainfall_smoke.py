import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data


GRAPH_SUBSET_PKL = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_graph_subset_train_v01"
    / "stockbridge_train_graph_subset_003.pkl"
)
SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"

EXPECTED_EVENT_IDS = ["R001", "R003", "R004"]
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


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the multi-event smoke loop."""
    parser = argparse.ArgumentParser(
        description="Run a tiny rainfall-forced SWE-GNN loop on three train events."
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
        help="Allow the full-graph multi-event loop to run on CPU.",
    )
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--start-time-index", type=int, default=2)
    parser.add_argument("--end-time-index", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rainfall-scale", type=float, default=300000.0)
    parser.add_argument("--shuffle", action="store_true")
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
            "CUDA is not available. Refusing to run full-graph multi-event "
            "rainfall tiny training loop on CPU unless --allow-cpu is set."
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


def validate_graph(graph: Data, expected_event_id: str) -> None:
    """Validate one graph from the prepared training subset."""
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
    assert graph.split == "train", (
        f"Expected {expected_event_id} split=train, got {graph.split}"
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
    """Evaluate all selected event/timestep pairs."""
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


def main() -> None:
    args = parse_args()

    print("=== Stockbridge SWE-GNN multi-event rainfall tiny training smoke ===")
    print("Graph subset PKL:", GRAPH_SUBSET_PKL)
    print("Original SWE-GNN repo:", SWEGNN_REPO)

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
    assert GRAPH_SUBSET_PKL.exists(), (
        f"Graph subset pickle not found: {GRAPH_SUBSET_PKL}"
    )
    assert SWEGNN_REPO.is_dir(), f"Original SWE-GNN repo not found: {SWEGNN_REPO}"

    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)

    from models.gnn import GNN

    with GRAPH_SUBSET_PKL.open("rb") as file:
        graphs = pickle.load(file)

    assert isinstance(graphs, list), f"Expected list, got {type(graphs).__name__}"
    assert len(graphs) == 3, f"Expected 3 graphs, got {len(graphs)}"

    for graph, expected_event_id in zip(graphs, EXPECTED_EVENT_IDS):
        validate_graph(graph, expected_event_id)

    actual_event_ids = [graph.event_id for graph in graphs]
    assert actual_event_ids == EXPECTED_EVENT_IDS, (
        f"Expected event order {EXPECTED_EVENT_IDS}, got {actual_event_ids}"
    )

    if args.start_time_index < 0:
        raise ValueError("--start-time-index must be greater than or equal to zero")
    if args.end_time_index >= int(graphs[0].nt) - 1:
        raise ValueError(
            f"--end-time-index must be less than {int(graphs[0].nt) - 1}; "
            f"got {args.end_time_index}"
        )
    if args.start_time_index > args.end_time_index:
        raise ValueError("--start-time-index must not exceed --end-time-index")

    time_indices = list(range(args.start_time_index, args.end_time_index + 1))
    event_time_pairs = [
        (event_index, t)
        for event_index in range(len(graphs))
        for t in time_indices
    ]

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
        assert tuple(edge_attr.shape) == EXPECTED_EDGE_ATTR_SHAPE
        assert_finite(f"{graph.event_id} slope_x", slope_x)
        assert_finite(f"{graph.event_id} slope_y", slope_y)
        assert_finite(f"{graph.event_id} edge_attr", edge_attr)
        prepared_events.append((graph, slope_x, slope_y, edge_attr))

    print("Event IDs:", actual_event_ids)
    print("Selected timesteps:", time_indices)
    print("Event/timestep pairs:", len(event_time_pairs))
    print("Input features:", INPUT_FEATURES)
    print("Rainfall scale:", args.rainfall_scale)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    node_features = 7
    edge_features = 3
    previous_t = 1
    model = GNN(
        node_features=node_features,
        edge_features=edge_features,
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
    print("model class:", model.__class__.__name__)
    print("type_GNN: SWEGNN")
    print("node_features:", node_features)
    print("edge_features:", edge_features)
    print("previous_t:", previous_t)
    print("hid_features:", args.hid_features)
    print("K:", args.K)
    print("epochs:", args.epochs)
    print("shuffle:", args.shuffle)
    print("parameter count:", parameter_count)

    print("\nOptimizer configuration:")
    print("optimizer: Adam")
    print("learning rate:", args.lr)
    print("weight decay:", args.weight_decay)

    initial_mean, initial_min, initial_max = evaluate_model(
        model,
        prepared_events,
        event_time_pairs,
        args.rainfall_scale,
        device,
        criterion,
    )
    print("\nInitial evaluation over all pairs:")
    print(f"mean loss: {initial_mean:.8g}")
    print(f"min loss: {initial_min:.8g}")
    print(f"max loss: {initial_max:.8g}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_pairs = event_time_pairs.copy()
        if args.shuffle:
            random.shuffle(epoch_pairs)

        epoch_losses = []
        model.train()

        for event_index, t in epoch_pairs:
            graph, slope_x, slope_y, edge_attr = prepared_events[event_index]
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
            assert_finite(f"{graph.event_id} training prediction at T{t}", pred)

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
                f"No model parameter received a gradient for {graph.event_id} at T{t}"
            )
            gradients_are_finite = all(
                torch.isfinite(parameter.grad).all().item()
                for parameter in parameters_with_gradients
            )
            assert gradients_are_finite, (
                f"Non-finite gradient for {graph.event_id} at T{t}"
            )

            optimizer.step()
            loss_value = loss.item()
            epoch_losses.append(loss_value)

            input_h_max = sample.x[:, INPUT_FEATURES.index("H_t")].max().item()
            target_h_max = sample.y[:, 0].max().item()
            target_qmag_max = sample.y[:, 1].max().item()
            rainfall_input_raw = sample.rainfall_input_raw.item()
            rainfall_target_raw = sample.rainfall_target_raw.item()
            rainfall_input_scaled = sample.rainfall_input_scaled.item()
            rainfall_target_scaled = sample.rainfall_target_scaled.item()
            print(
                f"epoch={epoch} event_id={graph.event_id} "
                f"t={t} target_t={t + 1} loss={loss_value:.8g} "
                f"input_H_max={input_h_max:.8g} "
                f"target_H_max={target_h_max:.8g} "
                f"target_Qmag_max={target_qmag_max:.8g} "
                f"rainfall_raw_input={rainfall_input_raw:.8g} "
                f"rainfall_raw_target={rainfall_target_raw:.8g} "
                f"rainfall_scaled_input={rainfall_input_scaled:.8g} "
                f"rainfall_scaled_target={rainfall_target_scaled:.8g}"
            )

        epoch_mean, epoch_min, epoch_max = summarize_losses(epoch_losses)
        epoch_elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch} summary: average_loss={epoch_mean:.8g} "
            f"min_loss={epoch_min:.8g} max_loss={epoch_max:.8g} "
            f"elapsed_seconds={epoch_elapsed:.2f}"
        )

    final_mean, final_min, final_max = evaluate_model(
        model,
        prepared_events,
        event_time_pairs,
        args.rainfall_scale,
        device,
        criterion,
    )

    first_parameter_absolute_change = torch.sum(
        torch.abs(first_parameter.detach() - first_parameter_before)
    ).item()
    assert first_parameter_absolute_change > 0.0, (
        f"First trainable parameter did not change: {first_parameter_name}"
    )

    print("\nFinal evaluation over all pairs:")
    print(f"mean loss: {final_mean:.8g}")
    print(f"min loss: {final_min:.8g}")
    print(f"max loss: {final_max:.8g}")

    print("\nParameter update:")
    print("first trainable parameter:", first_parameter_name)
    print(
        "first parameter absolute change: "
        f"{first_parameter_absolute_change:.8g}"
    )

    if device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print("\nCUDA diagnostics:")
        print("device name:", torch.cuda.get_device_name(device))
        print(f"max memory allocated: {max_memory_mb:.2f} MB")

    print(
        "\nStockbridge SWE-GNN multi-event rainfall tiny training loop "
        "smoke test passed."
    )


if __name__ == "__main__":
    main()
