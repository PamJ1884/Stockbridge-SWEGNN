import argparse
import pickle
import sys
from pathlib import Path

import torch


TEMPORAL_DIR = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_temporal_smoke_v01"
)
TEMPORAL_PKL = TEMPORAL_DIR / "R019_stockbridge_swegnn_temporal_smoke.pkl"

SWEGNN_REPO = Path.home() / "flood/01_repositories/SWE-GNN-paper-repository"

EXPECTED_X_SHAPE = (99430, 5)
EXPECTED_Y_SHAPE = (99430, 2, 24)
EXPECTED_EDGE_INDEX_SHAPE = (2, 396458)
EXPECTED_EDGE_ATTR_SHAPE = (396458, 3)
EXPECTED_PREDICTION_SHAPE = (99430, 2)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the one-forward-pass smoke test."""
    parser = argparse.ArgumentParser(
        description="Run one original SWE-GNN forward pass on the R019 smoke sample."
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
        help="Allow the full-graph forward pass to run on CPU.",
    )
    parser.add_argument("--hid-features", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
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
            "CUDA is not available. Refusing to run full-graph forward on CPU "
            "unless --allow-cpu is set."
        )

    return torch.device(device_type)


def print_min_max(name: str, tensor: torch.Tensor) -> None:
    """Print the minimum and maximum values in a tensor."""
    print(
        f"{name:20s} min={tensor.min().item():.8g} "
        f"max={tensor.max().item():.8g}"
    )


def main() -> None:
    args = parse_args()

    print("=== Stockbridge SWE-GNN one-forward temporal smoke test ===")
    print("Temporal PKL:", TEMPORAL_PKL)
    print("Original SWE-GNN repo:", SWEGNN_REPO)

    device = select_device(args.device, args.allow_cpu)
    print("Selected device:", device)

    assert args.hid_features > 0, "--hid-features must be greater than zero"
    assert args.K > 0, "--K must be greater than zero"
    assert TEMPORAL_PKL.exists(), f"Temporal pickle not found: {TEMPORAL_PKL}"
    assert SWEGNN_REPO.is_dir(), f"Original SWE-GNN repo not found: {SWEGNN_REPO}"

    swegnn_repo_path = str(SWEGNN_REPO)
    if swegnn_repo_path not in sys.path:
        sys.path.insert(0, swegnn_repo_path)

    from models.gnn import GNN

    with TEMPORAL_PKL.open("rb") as file:
        dataset = pickle.load(file)

    assert isinstance(dataset, list), f"Expected a list, got {type(dataset).__name__}"
    assert len(dataset) == 1, f"Expected one temporal sample, got {len(dataset)}"

    sample = dataset[0]
    assert tuple(sample.x.shape) == EXPECTED_X_SHAPE, (
        f"Expected x shape {EXPECTED_X_SHAPE}, got {tuple(sample.x.shape)}"
    )
    assert tuple(sample.y.shape) == EXPECTED_Y_SHAPE, (
        f"Expected y shape {EXPECTED_Y_SHAPE}, got {tuple(sample.y.shape)}"
    )
    assert tuple(sample.edge_index.shape) == EXPECTED_EDGE_INDEX_SHAPE, (
        "Expected edge_index shape "
        f"{EXPECTED_EDGE_INDEX_SHAPE}, got {tuple(sample.edge_index.shape)}"
    )
    assert tuple(sample.edge_attr.shape) == EXPECTED_EDGE_ATTR_SHAPE, (
        "Expected edge_attr shape "
        f"{EXPECTED_EDGE_ATTR_SHAPE}, got {tuple(sample.edge_attr.shape)}"
    )

    print("\nLoaded temporal sample summary:")
    print(sample)

    node_features = int(sample.x.shape[1])
    edge_features = int(sample.edge_attr.shape[1])
    previous_t = int(getattr(sample, "previous_t", 1))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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

    print("\nModel configuration:")
    print("model class:", model.__class__.__name__)
    print("node_features:", node_features)
    print("edge_features:", edge_features)
    print("previous_t:", previous_t)
    print("hid_features:", args.hid_features)
    print("K:", args.K)
    print("parameter count:", parameter_count)

    model.eval()
    with torch.no_grad():
        model_output = model(sample)

    if isinstance(model_output, (tuple, list)):
        assert model_output, "Model returned an empty tuple/list"
        print("Model returned a tuple/list; using its first element as pred.")
        pred = model_output[0]
    else:
        pred = model_output

    assert isinstance(pred, torch.Tensor), (
        f"Expected prediction tensor, got {type(pred).__name__}"
    )
    assert tuple(pred.shape) == EXPECTED_PREDICTION_SHAPE, (
        "Expected prediction shape "
        f"{EXPECTED_PREDICTION_SHAPE}, got {tuple(pred.shape)}"
    )
    assert not torch.isnan(pred).any().item(), "Prediction contains NaN values"
    assert not torch.isinf(pred).any().item(), "Prediction contains infinite values"

    target = sample.y[:, :, 0]
    assert tuple(target.shape) == EXPECTED_PREDICTION_SHAPE, (
        f"Expected target shape {EXPECTED_PREDICTION_SHAPE}, got {tuple(target.shape)}"
    )

    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()

    print("\nForward-pass shapes:")
    print("pred:", tuple(pred.shape))
    print("target:", tuple(target.shape))

    print("\nDiagnostics:")
    print(f"MSE: {mse:.8g}")
    print(f"MAE: {mae:.8g}")
    print_min_max("predicted H", pred[:, 0])
    print_min_max("predicted Qmag", pred[:, 1])
    print_min_max("target H_T1", target[:, 0])
    print_min_max("target Qmag_T1", target[:, 1])

    if device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print("\nCUDA diagnostics:")
        print("device name:", torch.cuda.get_device_name(device))
        print(f"max memory allocated: {max_memory_mb:.2f} MB")

    print("\nStockbridge SWE-GNN one-forward smoke test passed.")


if __name__ == "__main__":
    main()
