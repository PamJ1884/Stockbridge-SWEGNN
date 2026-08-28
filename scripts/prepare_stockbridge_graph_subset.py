import argparse
import json
import pickle
import re
from pathlib import Path

import torch
from torch_geometric.data import Data


INPUT_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2_core_v01"
)
OUTPUT_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_graph_subset_train_v01"
)

EXPECTED_TENSOR_SHAPE = (305, 326, 25, 5)
EXPECTED_NUM_NODES = 99430
EXPECTED_NUM_EDGES = 396458
CELL_SIZE = 1.0

CHANNEL_ORDER = {
    "H": 0,
    "Vx": 1,
    "Vy": 2,
    "rainfall": 3,
    "DEM": 4,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for graph subset conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a small Stockbridge event subset to PyG graphs."
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--event-ids", default="R001,R003,R004")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_event_ids(value: str) -> list[str]:
    """Parse and validate ordered, comma-separated Stockbridge event IDs."""
    event_ids = [event_id.strip() for event_id in value.split(",")]
    if not event_ids or any(not event_id for event_id in event_ids):
        raise ValueError("--event-ids must contain one or more comma-separated IDs")

    invalid = [
        event_id
        for event_id in event_ids
        if re.fullmatch(r"R\d{3}", event_id) is None
    ]
    if invalid:
        raise ValueError(f"Invalid event IDs: {invalid}; expected values such as R001")

    return event_ids


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Assert that a tensor contains neither NaN nor infinite values."""
    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN/inf values"


def print_min_max(name: str, tensor: torch.Tensor) -> None:
    """Print the minimum and maximum values in a tensor."""
    print(
        f"{name:12s} min={tensor.min().item():.8g} "
        f"max={tensor.max().item():.8g}"
    )


def build_grid_edges(
    ny: int,
    nx: int,
    cell_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create directed 4-neighbour edges and grid-unit geometry."""
    node_ids = torch.arange(ny * nx, dtype=torch.long).reshape(ny, nx)

    left = node_ids[:, :-1].reshape(-1)
    right = node_ids[:, 1:].reshape(-1)
    top = node_ids[:-1, :].reshape(-1)
    bottom = node_ids[1:, :].reshape(-1)

    row = torch.cat([left, right, top, bottom])
    col = torch.cat([right, left, bottom, top])
    edge_index = torch.stack([row, col], dim=0)

    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    pos = torch.stack(
        [xx.reshape(-1) * cell_size, yy.reshape(-1) * cell_size],
        dim=1,
    )

    edge_relative_distance = pos[col] - pos[row]
    edge_distance = torch.linalg.norm(edge_relative_distance, dim=1)

    assert tuple(edge_index.shape) == (2, EXPECTED_NUM_EDGES), (
        "Expected edge_index shape "
        f"(2, {EXPECTED_NUM_EDGES}), got {tuple(edge_index.shape)}"
    )
    assert tuple(pos.shape) == (EXPECTED_NUM_NODES, 2), (
        f"Expected pos shape ({EXPECTED_NUM_NODES}, 2), got {tuple(pos.shape)}"
    )
    assert not (row == col).any().item(), "Grid connectivity contains self-loops"
    assert_finite("pos", pos)
    assert_finite("edge_relative_distance", edge_relative_distance)
    assert_finite("edge_distance", edge_distance)
    assert (edge_distance > 0).all().item(), "Edge distance must be positive"

    return edge_index, pos, edge_relative_distance, edge_distance


def tensor_statistics(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    """Return JSON-compatible minimum, maximum, and mean statistics."""
    return {
        f"{prefix}_min": tensor.min().item(),
        f"{prefix}_max": tensor.max().item(),
        f"{prefix}_mean": tensor.mean().item(),
    }


def convert_event(
    event_id: str,
    split: str,
    source_path: Path,
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    edge_relative_distance: torch.Tensor,
    edge_distance: torch.Tensor,
) -> tuple[Data, dict]:
    """Convert one Stockbridge tensor event to a PyG Data graph."""
    tensor = torch.load(source_path, map_location="cpu")
    assert isinstance(tensor, torch.Tensor), (
        f"Expected torch.Tensor for {event_id}, got {type(tensor).__name__}"
    )
    assert tuple(tensor.shape) == EXPECTED_TENSOR_SHAPE, (
        f"Expected {event_id} shape {EXPECTED_TENSOR_SHAPE}, "
        f"got {tuple(tensor.shape)}"
    )
    assert tensor.is_floating_point(), (
        f"Expected floating-point tensor for {event_id}, got {tensor.dtype}"
    )

    ny, nx, nt, _ = tensor.shape
    num_nodes = ny * nx
    assert num_nodes == EXPECTED_NUM_NODES, (
        f"Expected {EXPECTED_NUM_NODES} nodes, got {num_nodes}"
    )

    H = (
        tensor[..., CHANNEL_ORDER["H"]]
        .reshape(num_nodes, nt)
        .contiguous()
        .float()
    )
    Vx = (
        tensor[..., CHANNEL_ORDER["Vx"]]
        .reshape(num_nodes, nt)
        .contiguous()
        .float()
    )
    Vy = (
        tensor[..., CHANNEL_ORDER["Vy"]]
        .reshape(num_nodes, nt)
        .contiguous()
        .float()
    )
    rainfall = (
        tensor[..., CHANNEL_ORDER["rainfall"]]
        .reshape(num_nodes, nt)
        .contiguous()
        .float()
    )
    DEM = (
        tensor[:, :, 0, CHANNEL_ORDER["DEM"]]
        .reshape(num_nodes)
        .contiguous()
        .float()
    )
    rainfall_global = (
        tensor[0, 0, :, CHANNEL_ORDER["rainfall"]]
        .clone()
        .contiguous()
        .float()
    )

    Qmag = torch.sqrt((Vx * H) ** 2 + (Vy * H) ** 2)
    row, col = edge_index
    edge_slope = (DEM[col] - DEM[row]) / edge_distance

    rainfall_spatial_max_abs_diff = torch.abs(
        rainfall - rainfall_global.unsqueeze(0)
    ).max()

    finite_tensors = {
        "DEM": DEM,
        "H": H,
        "Vx": Vx,
        "Vy": Vy,
        "rainfall": rainfall,
        "rainfall_global": rainfall_global,
        "Qmag": Qmag,
        "pos": pos,
        "edge_distance": edge_distance,
        "edge_relative_distance": edge_relative_distance,
        "edge_slope": edge_slope,
    }
    for name, value in finite_tensors.items():
        assert_finite(f"{event_id} {name}", value)
    assert torch.isfinite(rainfall_spatial_max_abs_diff).item(), (
        f"{event_id} rainfall spatial difference is NaN/inf"
    )

    data = Data()
    data.edge_index = edge_index
    data.edge_distance = edge_distance.float()
    data.edge_relative_distance = edge_relative_distance.float()
    data.edge_slope = edge_slope.float()
    data.num_nodes = num_nodes
    data.pos = pos.float()
    data.DEM = DEM
    data.H = H
    data.WD = H
    data.Vx = Vx
    data.Vy = Vy
    data.VX = Vx
    data.VY = Vy
    data.Qmag = Qmag.float()
    data.rainfall = rainfall
    data.rainfall_global = rainfall_global
    data.ny = ny
    data.nx = nx
    data.nt = nt
    data.cell_size = CELL_SIZE
    data.event_id = event_id
    data.split = split
    data.source_pt_file = str(source_path)

    event_manifest = {
        "event_id": event_id,
        "source_pt_file": str(source_path),
        "tensor_shape": list(tensor.shape),
        **tensor_statistics("H", H),
        **tensor_statistics("Qmag", Qmag),
        **tensor_statistics("rainfall", rainfall),
        "rainfall_global": rainfall_global.tolist(),
        "rainfall_spatial_max_abs_diff": (
            rainfall_spatial_max_abs_diff.item()
        ),
        **tensor_statistics("DEM", DEM),
    }

    print(f"\nConverted {event_id} from {source_path}")
    print("tensor shape:", tuple(tensor.shape), "dtype:", tensor.dtype)
    print("H shape:", tuple(H.shape))
    print("Vx shape:", tuple(Vx.shape), "Vy shape:", tuple(Vy.shape))
    print("Qmag shape:", tuple(Qmag.shape))
    print("rainfall shape:", tuple(rainfall.shape))
    print("DEM shape:", tuple(DEM.shape))
    print("pos shape:", tuple(pos.shape))
    print(
        "edge shapes:",
        tuple(edge_index.shape),
        tuple(edge_relative_distance.shape),
        tuple(edge_slope.shape),
    )
    print_min_max("H", H)
    print_min_max("Qmag", Qmag)
    print_min_max("rainfall", rainfall)
    print_min_max("DEM", DEM)
    print(
        "rainfall spatial max abs difference: "
        f"{rainfall_spatial_max_abs_diff.item():.8g}"
    )

    return data, event_manifest


def main() -> None:
    args = parse_args()
    event_ids = parse_event_ids(args.event_ids)

    if re.fullmatch(r"[A-Za-z0-9_-]+", args.split) is None:
        raise ValueError(f"Invalid split name: {args.split!r}")

    input_paths = [
        INPUT_ROOT / args.split / f"{event_id}_stockbridge_T000_T024.pt"
        for event_id in event_ids
    ]
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        formatted_paths = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing input tensor files:\n{formatted_paths}")

    n_events = len(event_ids)
    output_stem = f"stockbridge_{args.split}_graph_subset_{n_events:03d}"
    output_pkl = args.output_dir / f"{output_stem}.pkl"
    output_json = args.output_dir / f"{output_stem}.json"

    existing_outputs = [
        path for path in (output_pkl, output_json) if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        formatted_paths = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing output files without --overwrite:\n"
            f"{formatted_paths}"
        )

    print("=== Prepare Stockbridge SWE-GNN train graph subset ===")
    print("Selected split:", args.split)
    print("Selected event IDs:", event_ids)
    print("Output directory:", args.output_dir)

    ny, nx, _, _ = EXPECTED_TENSOR_SHAPE
    edge_index, pos, edge_relative_distance, edge_distance = build_grid_edges(
        ny=ny,
        nx=nx,
        cell_size=CELL_SIZE,
    )

    graphs = []
    event_manifests = []
    for event_id, source_path in zip(event_ids, input_paths):
        graph, event_manifest = convert_event(
            event_id=event_id,
            split=args.split,
            source_path=source_path,
            edge_index=edge_index,
            pos=pos,
            edge_relative_distance=edge_relative_distance,
            edge_distance=edge_distance,
        )
        graphs.append(graph)
        event_manifests.append(event_manifest)

    manifest = {
        "script_name": Path(__file__).name,
        "split": args.split,
        "event_ids": event_ids,
        "n_events": n_events,
        "input_root": str(INPUT_ROOT),
        "output_pkl": str(output_pkl),
        "output_json": str(output_json),
        "expected_tensor_shape": list(EXPECTED_TENSOR_SHAPE),
        "num_nodes": EXPECTED_NUM_NODES,
        "num_edges": EXPECTED_NUM_EDGES,
        "cell_size": CELL_SIZE,
        "channel_order": CHANNEL_ORDER,
        "events": event_manifests,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as file:
        pickle.dump(graphs, file)
    with output_json.open("w") as file:
        json.dump(manifest, file, indent=2)

    print("\nSaved graph subset:")
    print(output_pkl)
    print(output_json)
    print("\nStockbridge SWE-GNN train graph subset preparation passed.")


if __name__ == "__main__":
    main()
