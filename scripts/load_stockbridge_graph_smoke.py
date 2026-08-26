from pathlib import Path
import pickle

import torch
from torch_geometric.data import Data


GRAPH_DIR = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_graph_smoke_v01"
)
PKL_PATH = GRAPH_DIR / "R019_stockbridge_swegnn_graph_smoke.pkl"
JSON_PATH = GRAPH_DIR / "R019_stockbridge_swegnn_graph_smoke.json"


EXPECTED_ATTRIBUTES = (
    "edge_index",
    "edge_distance",
    "edge_relative_distance",
    "edge_slope",
    "pos",
    "DEM",
    "H",
    "WD",
    "Vx",
    "Vy",
    "VX",
    "VY",
    "Qmag",
    "rainfall",
    "rainfall_global",
    "ny",
    "nx",
    "nt",
    "num_nodes",
    "event_id",
    "split",
)

EXPECTED_SHAPES = {
    "edge_index": (2, 396458),
    "edge_distance": (396458,),
    "edge_relative_distance": (396458, 2),
    "edge_slope": (396458,),
    "pos": (99430, 2),
    "DEM": (99430,),
    "H": (99430, 25),
    "WD": (99430, 25),
    "Vx": (99430, 25),
    "Vy": (99430, 25),
    "VX": (99430, 25),
    "VY": (99430, 25),
    "Qmag": (99430, 25),
    "rainfall": (99430, 25),
    "rainfall_global": (25,),
}


def print_min_max(name: str, tensor: torch.Tensor) -> None:
    """Print the minimum and maximum values in a tensor."""
    print(
        f"{name:20s} min={tensor.min().item():.8g} "
        f"max={tensor.max().item():.8g}"
    )


def main() -> None:
    print("=== Load Stockbridge SWE-GNN graph smoke ===")
    print("PKL path:", PKL_PATH)
    print("JSON path:", JSON_PATH)
    print("PKL exists:", PKL_PATH.exists())
    print("JSON exists:", JSON_PATH.exists())

    assert PKL_PATH.exists(), f"Graph pickle not found: {PKL_PATH}"

    with PKL_PATH.open("rb") as file:
        dataset = pickle.load(file)

    assert isinstance(dataset, list), f"Expected a list, got {type(dataset).__name__}"
    assert len(dataset) == 1, f"Expected one graph, got {len(dataset)}"

    data = dataset[0]
    assert isinstance(data, Data), f"Expected PyG Data, got {type(data).__name__}"

    print("\nData object summary:")
    print(data)

    print("\nKeys and shapes:")
    keys = data.keys() if callable(data.keys) else data.keys
    for key in keys:
        value = getattr(data, key)
        if hasattr(value, "shape"):
            print(f"{key:24s} shape={tuple(value.shape)}")
        else:
            print(f"{key:24s} type={type(value).__name__} value={value}")

    missing = [name for name in EXPECTED_ATTRIBUTES if not hasattr(data, name)]
    assert not missing, f"Missing expected attributes: {missing}"

    assert data.ny == 305, f"Expected ny=305, got {data.ny}"
    assert data.nx == 326, f"Expected nx=326, got {data.nx}"
    assert data.nt == 25, f"Expected nt=25, got {data.nt}"
    assert data.num_nodes == 99430, (
        f"Expected num_nodes=99430, got {data.num_nodes}"
    )

    for name, expected_shape in EXPECTED_SHAPES.items():
        value = getattr(data, name)
        actual_shape = tuple(value.shape)
        assert actual_shape == expected_shape, (
            f"Expected {name} shape {expected_shape}, got {actual_shape}"
        )

    print("\nValue checks:")
    print_min_max("DEM", data.DEM)
    print_min_max("H", data.H)
    print_min_max("H at T0", data.H[:, 0])
    print_min_max("H at T24", data.H[:, 24])
    print_min_max("Vx", data.Vx)
    print_min_max("Vy", data.Vy)
    print_min_max("Qmag", data.Qmag)
    print_min_max("rainfall", data.rainfall)
    print("rainfall_global values:", data.rainfall_global.tolist())

    wd_matches_h = torch.allclose(data.WD, data.H)
    vx_alias_matches = torch.allclose(data.VX, data.Vx)
    vy_alias_matches = torch.allclose(data.VY, data.Vy)
    expected_qmag = torch.sqrt(
        (data.Vx * data.H) ** 2 + (data.Vy * data.H) ** 2
    )
    qmag_matches = torch.allclose(
        data.Qmag,
        expected_qmag,
        rtol=1e-5,
        atol=1e-6,
    )

    print("\nConsistency checks:")
    print("WD equals H:", wd_matches_h)
    print("VX equals Vx:", vx_alias_matches)
    print("VY equals Vy:", vy_alias_matches)
    print("Qmag matches velocity-depth calculation:", qmag_matches)

    assert wd_matches_h, "WD does not match H"
    assert vx_alias_matches, "VX does not match Vx"
    assert vy_alias_matches, "VY does not match Vy"
    assert qmag_matches, "Qmag does not match the velocity-depth calculation"

    source, target = data.edge_index
    num_self_loops = int((source == target).sum().item())

    print("\nEdge sanity checks:")
    print("Number of self-loops:", num_self_loops)
    print_min_max("edge_distance", data.edge_distance)
    print_min_max("edge_slope", data.edge_slope)
    print("First 5 edges:")
    for edge_number, (src, dst) in enumerate(
        data.edge_index[:, :5].t().tolist(),
        start=1,
    ):
        print(f"  {edge_number}: {src} -> {dst}")

    assert num_self_loops == 0, f"Expected no self-loops, got {num_self_loops}"

    print("\nStockbridge SWE-GNN graph smoke load test passed.")


if __name__ == "__main__":
    main()
