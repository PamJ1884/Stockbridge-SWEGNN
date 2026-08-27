import argparse
import pickle
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data


GRAPH_DIR = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_graph_smoke_v01"
)
GRAPH_PKL = GRAPH_DIR / "R019_stockbridge_swegnn_graph_smoke.pkl"

SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"

EXPECTED_NUM_NODES = 99430
EXPECTED_X_SHAPE = (99430, 5)
EXPECTED_TARGET_SHAPE = (99430, 2)
EXPECTED_EDGE_INDEX_SHAPE = (2, 396458)
EXPECTED_EDGE_ATTR_SHAPE = (396458, 3)

INPUT_FEATURES = ["slope_x", "slope_y", "DEM", "H_t", "Qmag_t"]
EDGE_FEATURES = ["dx", "dy", "edge_slope"]
OUTPUT_VARIABLES = ["H", "Qmag"]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the one-step training smoke test."""
    parser = argparse.ArgumentParser(
        description="Run one SWE-GNN optimizer update at an R019 wet timestep."
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
        help="Allow the full-graph training step to run on CPU.",
    )
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--input-time-index", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    return parser.parse_args()


def select_device(requested_device: str, allow_cpu: bool) -> torch.device:
    """Resolve the requested device and protect against accidental CPU runs."""
    if requested_device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = requested_device

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    if device_type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Refusing to run full-graph training step on CPU "
            "unless --allow-cpu is set."
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
    assert not torch.isnan(tensor).any().item(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any().item(), f"{name} contains infinite values"


def extract_prediction(model_output, phase: str):
    """Extract a prediction tensor from the model's possible output formats."""
    if isinstance(model_output, (tuple, list)):
        assert model_output, f"Model returned an empty tuple/list {phase}"
        print(f"Model returned a tuple/list {phase}; using its first element.")
        return model_output[0]
    return model_output


def print_min_max(name: str, tensor: torch.Tensor) -> None:
    """Print the minimum and maximum values in a tensor."""
    print(
        f"{name:24s} min={tensor.min().item():.8g} "
        f"max={tensor.max().item():.8g}"
    )


def main() -> None:
    args = parse_args()

    print("=== Stockbridge SWE-GNN one-step training smoke test ===")
    print("Graph PKL:", GRAPH_PKL)
    print("Original SWE-GNN repo:", SWEGNN_REPO)

    device = select_device(args.device, args.allow_cpu)
    print("Selected device:", device)

    assert args.hid_features > 0, "--hid-features must be greater than zero"
    assert args.K > 0, "--K must be greater than zero"
    assert args.lr > 0.0, "--lr must be greater than zero"
    assert args.weight_decay >= 0.0, "--weight-decay must not be negative"
    assert GRAPH_PKL.exists(), f"Graph pickle not found: {GRAPH_PKL}"
    assert SWEGNN_REPO.is_dir(), f"Original SWE-GNN repo not found: {SWEGNN_REPO}"

    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)

    from models.gnn import GNN

    with GRAPH_PKL.open("rb") as file:
        dataset = pickle.load(file)

    assert isinstance(dataset, list), f"Expected a list, got {type(dataset).__name__}"
    assert len(dataset) == 1, f"Expected one graph, got {len(dataset)}"

    graph = dataset[0]
    assert isinstance(graph, Data), f"Expected PyG Data, got {type(graph).__name__}"

    required_attributes = (
        "H",
        "Qmag",
        "DEM",
        "edge_index",
        "edge_relative_distance",
        "edge_slope",
        "rainfall_global",
        "pos",
        "event_id",
        "split",
        "ny",
        "nx",
        "nt",
        "num_nodes",
    )
    missing = [name for name in required_attributes if not hasattr(graph, name)]
    assert not missing, f"Missing expected graph attributes: {missing}"

    expected_graph_shapes = {
        "H": (99430, 25),
        "Qmag": (99430, 25),
        "DEM": (99430,),
        "edge_index": EXPECTED_EDGE_INDEX_SHAPE,
        "edge_relative_distance": (396458, 2),
        "edge_slope": (396458,),
        "rainfall_global": (25,),
        "pos": (99430, 2),
    }
    for name, expected_shape in expected_graph_shapes.items():
        actual_shape = tuple(getattr(graph, name).shape)
        assert actual_shape == expected_shape, (
            f"Expected {name} shape {expected_shape}, got {actual_shape}"
        )

    assert graph.ny == 305, f"Expected ny=305, got {graph.ny}"
    assert graph.nx == 326, f"Expected nx=326, got {graph.nx}"
    assert graph.nt == 25, f"Expected nt=25, got {graph.nt}"
    assert graph.num_nodes == EXPECTED_NUM_NODES, (
        f"Expected num_nodes={EXPECTED_NUM_NODES}, got {graph.num_nodes}"
    )

    t = args.input_time_index
    if t < 0 or t >= int(graph.nt) - 1:
        raise ValueError(
            f"--input-time-index must satisfy 0 <= index < {int(graph.nt) - 1}; "
            f"got {t}"
        )
    target_t = t + 1

    print("Input time index:", t)
    print("Target time index:", target_t)

    slope_x, slope_y = calculate_dem_slopes(
        graph.DEM,
        ny=int(graph.ny),
        nx=int(graph.nx),
    )
    H_t = graph.H[:, t]
    Qmag_t = graph.Qmag[:, t]
    target = torch.stack(
        [graph.H[:, target_t], graph.Qmag[:, target_t]],
        dim=1,
    )
    x = torch.stack([slope_x, slope_y, graph.DEM, H_t, Qmag_t], dim=1)
    edge_attr = torch.cat(
        [graph.edge_relative_distance, graph.edge_slope.view(-1, 1)],
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
    sample.target_time_index = target_t
    sample.previous_t = 1
    sample.event_id = graph.event_id
    sample.split = graph.split
    sample.ny = graph.ny
    sample.nx = graph.nx
    sample.nt = graph.nt
    sample.num_nodes = graph.num_nodes
    sample.rainfall_input = graph.rainfall_global[t].float()
    sample.rainfall_target = graph.rainfall_global[target_t].float()
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

    print("\nIn-memory timestep sample summary:")
    print(sample)

    node_features = int(sample.x.shape[1])
    edge_features = int(sample.edge_attr.shape[1])
    previous_t = int(sample.previous_t)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(args.seed)
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
    sample = sample.to(device)

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
    print("parameter count:", parameter_count)

    print("\nOptimizer configuration:")
    print("optimizer: Adam")
    print("learning rate:", args.lr)
    print("weight decay:", args.weight_decay)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = extract_prediction(model(sample), "before the update")

    assert isinstance(pred, torch.Tensor), (
        f"Expected prediction tensor, got {type(pred).__name__}"
    )
    assert tuple(pred.shape) == EXPECTED_TARGET_SHAPE, (
        f"Expected pred shape {EXPECTED_TARGET_SHAPE}, got {tuple(pred.shape)}"
    )
    assert_finite("pred before update", pred)

    loss = criterion(pred, sample.y)
    assert torch.isfinite(loss).item(), "Loss before update is not finite"
    loss_before = loss.item()
    loss.backward()

    parameters_with_gradients = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    num_parameters_with_gradients = len(parameters_with_gradients)
    assert num_parameters_with_gradients > 0, "No model parameter received a gradient"

    has_nonfinite_gradient = any(
        not torch.isfinite(parameter.grad).all().item()
        for parameter in parameters_with_gradients
    )
    gradient_norms = torch.stack(
        [parameter.grad.detach().norm(2) for parameter in parameters_with_gradients]
    )
    total_gradient_norm = torch.linalg.vector_norm(gradient_norms).item()
    max_parameter_gradient_norm = gradient_norms.max().item()

    assert not has_nonfinite_gradient, "At least one gradient contains NaN/inf values"

    optimizer.step()

    first_parameter_absolute_change = torch.sum(
        torch.abs(first_parameter.detach() - first_parameter_before)
    ).item()
    assert first_parameter_absolute_change > 0.0, (
        f"First trainable parameter did not change: {first_parameter_name}"
    )

    with torch.no_grad():
        pred_after = extract_prediction(model(sample), "after the update")
        assert isinstance(pred_after, torch.Tensor), (
            f"Expected post-update prediction tensor, got {type(pred_after).__name__}"
        )
        assert tuple(pred_after.shape) == EXPECTED_TARGET_SHAPE, (
            "Expected post-update pred shape "
            f"{EXPECTED_TARGET_SHAPE}, got {tuple(pred_after.shape)}"
        )
        assert_finite("pred after update", pred_after)
        loss_after_tensor = criterion(pred_after, sample.y)

    assert torch.isfinite(loss_after_tensor).item(), "Loss after update is not finite"
    loss_after = loss_after_tensor.item()

    print("\nShapes:")
    print("input x:", tuple(sample.x.shape))
    print("target y:", tuple(sample.y.shape))
    print("edge_index:", tuple(sample.edge_index.shape))
    print("edge_attr:", tuple(sample.edge_attr.shape))
    print("pred before update:", tuple(pred.shape))
    print("pred after update:", tuple(pred_after.shape))

    print("\nValue diagnostics:")
    print_min_max(f"input H_T{t}", sample.x[:, 3])
    print_min_max(f"input Qmag_T{t}", sample.x[:, 4])
    print_min_max(f"target H_T{target_t}", sample.y[:, 0])
    print_min_max(f"target Qmag_T{target_t}", sample.y[:, 1])
    print_min_max("predicted H before", pred[:, 0])
    print_min_max("predicted Qmag before", pred[:, 1])
    print_min_max("predicted H after", pred_after[:, 0])
    print_min_max("predicted Qmag after", pred_after[:, 1])
    print(f"rainfall_input: {sample.rainfall_input.item():.8g}")
    print(f"rainfall_target: {sample.rainfall_target.item():.8g}")

    print("\nUpdate diagnostics:")
    print(f"loss_before: {loss_before:.8g}")
    print(f"loss_after: {loss_after:.8g}")
    print("parameters with gradients:", num_parameters_with_gradients)
    print(f"total gradient norm: {total_gradient_norm:.8g}")
    print(f"max single-parameter gradient norm: {max_parameter_gradient_norm:.8g}")
    print("any gradient NaN/inf:", has_nonfinite_gradient)
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

    print("\nStockbridge SWE-GNN one-step training smoke test passed.")


if __name__ == "__main__":
    main()
