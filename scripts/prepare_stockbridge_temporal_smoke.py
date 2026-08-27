from pathlib import Path
import json
import pickle

import torch
from torch_geometric.data import Data


GRAPH_DIR = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_graph_smoke_v01"
)
GRAPH_PKL = GRAPH_DIR / "R019_stockbridge_swegnn_graph_smoke.pkl"

OUT_DIR = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_temporal_smoke_v01"
)
OUT_PKL = OUT_DIR / "R019_stockbridge_swegnn_temporal_smoke.pkl"
OUT_JSON = OUT_DIR / "R019_stockbridge_swegnn_temporal_smoke.json"

INPUT_FEATURES = ["slope_x", "slope_y", "DEM", "H_T0", "Qmag_T0"]
EDGE_FEATURES = ["dx", "dy", "edge_slope"]
OUTPUT_VARIABLES = ["H", "Qmag"]


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


def print_min_max(name: str, tensor: torch.Tensor) -> None:
    """Print the minimum and maximum values in a tensor."""
    print(
        f"{name:20s} min={tensor.min().item():.8g} "
        f"max={tensor.max().item():.8g}"
    )


def main() -> None:
    print("=== Prepare Stockbridge SWE-GNN temporal smoke ===")
    print("Input graph:", GRAPH_PKL)
    print("Input exists:", GRAPH_PKL.exists())
    print("Output PKL:", OUT_PKL)
    print("Output JSON:", OUT_JSON)

    assert GRAPH_PKL.exists(), f"Graph pickle not found: {GRAPH_PKL}"

    with GRAPH_PKL.open("rb") as file:
        dataset = pickle.load(file)

    assert isinstance(dataset, list), f"Expected a list, got {type(dataset).__name__}"
    assert len(dataset) == 1, f"Expected one graph, got {len(dataset)}"

    data = dataset[0]
    assert isinstance(data, Data), f"Expected PyG Data, got {type(data).__name__}"

    required_attributes = (
        "ny",
        "nx",
        "nt",
        "num_nodes",
        "edge_index",
        "edge_relative_distance",
        "edge_slope",
        "pos",
        "DEM",
        "H",
        "Qmag",
        "rainfall_global",
        "event_id",
        "split",
    )
    missing = [name for name in required_attributes if not hasattr(data, name)]
    assert not missing, f"Missing expected graph attributes: {missing}"

    assert data.ny == 305, f"Expected ny=305, got {data.ny}"
    assert data.nx == 326, f"Expected nx=326, got {data.nx}"
    assert data.nt == 25, f"Expected nt=25, got {data.nt}"
    assert data.num_nodes == 99430, (
        f"Expected num_nodes=99430, got {data.num_nodes}"
    )

    expected_input_shapes = {
        "edge_index": (2, 396458),
        "edge_relative_distance": (396458, 2),
        "edge_slope": (396458,),
        "pos": (99430, 2),
        "DEM": (99430,),
        "H": (99430, 25),
        "Qmag": (99430, 25),
        "rainfall_global": (25,),
    }
    for name, expected_shape in expected_input_shapes.items():
        actual_shape = tuple(getattr(data, name).shape)
        assert actual_shape == expected_shape, (
            f"Expected {name} shape {expected_shape}, got {actual_shape}"
        )

    print("\nLoaded graph summary:")
    print(data)

    ny = int(data.ny)
    nx = int(data.nx)
    slope_x, slope_y = calculate_dem_slopes(data.DEM, ny, nx)

    H_T0 = data.H[:, 0]
    Qmag_T0 = data.Qmag[:, 0]
    H_future = data.H[:, 1:25]
    Qmag_future = data.Qmag[:, 1:25]

    x = torch.stack([slope_x, slope_y, data.DEM, H_T0, Qmag_T0], dim=1)
    y = torch.stack([H_future, Qmag_future], dim=1)
    edge_attr = torch.cat(
        [data.edge_relative_distance, data.edge_slope.view(-1, 1)],
        dim=1,
    )

    sample = Data()
    sample.edge_index = data.edge_index
    sample.edge_attr = edge_attr.float()
    sample.x = x.float()
    sample.y = y.float()
    sample.pos = data.pos.float()
    sample.DEM = data.DEM.float()

    sample.rainfall_global = data.rainfall_global.float()
    sample.rainfall_future = data.rainfall_global[1:25].float()
    sample.input_time_index = 0
    sample.target_start_index = 1
    sample.target_end_index = 24
    sample.rollout_steps = 24
    sample.previous_t = 1
    sample.output_variables = OUTPUT_VARIABLES
    sample.input_features = INPUT_FEATURES
    sample.edge_features = EDGE_FEATURES
    sample.event_id = data.event_id
    sample.split = data.split
    sample.ny = data.ny
    sample.nx = data.nx
    sample.nt = data.nt
    sample.num_nodes = data.num_nodes
    sample.source_graph = str(GRAPH_PKL)

    assert tuple(sample.x.shape) == (99430, 5), (
        f"Expected x shape (99430, 5), got {tuple(sample.x.shape)}"
    )
    assert tuple(sample.y.shape) == (99430, 2, 24), (
        f"Expected y shape (99430, 2, 24), got {tuple(sample.y.shape)}"
    )
    assert tuple(sample.edge_attr.shape) == (396458, 3), (
        "Expected edge_attr shape (396458, 3), "
        f"got {tuple(sample.edge_attr.shape)}"
    )
    assert tuple(sample.rainfall_future.shape) == (24,), (
        "Expected rainfall_future shape (24,), "
        f"got {tuple(sample.rainfall_future.shape)}"
    )
    assert not torch.isnan(sample.x).any().item(), "x contains NaN values"
    assert not torch.isnan(sample.y).any().item(), "y contains NaN values"
    assert not torch.isnan(sample.edge_attr).any().item(), (
        "edge_attr contains NaN values"
    )

    print("\nCreated temporal sample summary:")
    print(sample)

    print("\nShapes:")
    print("x:", tuple(sample.x.shape))
    print("y:", tuple(sample.y.shape))
    print("edge_index:", tuple(sample.edge_index.shape))
    print("edge_attr:", tuple(sample.edge_attr.shape))
    print("rainfall_global:", tuple(sample.rainfall_global.shape))
    print("rainfall_future:", tuple(sample.rainfall_future.shape))

    print("\nInput feature ranges:")
    for column, feature_name in enumerate(INPUT_FEATURES):
        print_min_max(feature_name, sample.x[:, column])

    print("\nTarget ranges:")
    print_min_max("y[H]", sample.y[:, 0, :])
    print_min_max("y[Qmag]", sample.y[:, 1, :])

    print("\nRainfall ranges:")
    print_min_max("rainfall_global", sample.rainfall_global)
    print_min_max("rainfall_future", sample.rainfall_future)

    num_edges = int(sample.edge_index.shape[1])
    metadata = {
        "input_graph_path": str(GRAPH_PKL),
        "output_pkl_path": str(OUT_PKL),
        "event_id": sample.event_id,
        "split": sample.split,
        "ny": int(sample.ny),
        "nx": int(sample.nx),
        "nt": int(sample.nt),
        "num_nodes": int(sample.num_nodes),
        "num_edges": num_edges,
        "x_shape": list(sample.x.shape),
        "y_shape": list(sample.y.shape),
        "edge_attr_shape": list(sample.edge_attr.shape),
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "input_time_index": sample.input_time_index,
        "target_time_range": [sample.target_start_index, sample.target_end_index],
        "rollout_steps": sample.rollout_steps,
        "previous_t": sample.previous_t,
        "notes": [
            "This is a one-event temporal smoke sample for T0 to T1:T24 forecasting.",
            "Rainfall is stored as rainfall_global and rainfall_future but is not yet "
            "included in x for this smoke version.",
            "DEM slopes use finite differences in grid-cell units.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PKL.open("wb") as file:
        pickle.dump([sample], file)

    with OUT_JSON.open("w") as file:
        json.dump(metadata, file, indent=2)

    print("\nSaved:")
    print(OUT_PKL)
    print(OUT_JSON)
    print("\nStockbridge SWE-GNN temporal smoke preparation passed.")


if __name__ == "__main__":
    main()
