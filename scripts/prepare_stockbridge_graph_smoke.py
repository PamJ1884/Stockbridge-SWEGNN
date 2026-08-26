from pathlib import Path
import json
import pickle

import torch
from torch_geometric.data import Data


PT_ROOT = Path.home() / "flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2_core_v01"
EVENT_PATH = PT_ROOT / "test/R019_stockbridge_T000_T024.pt"

OUT_DIR = Path.home() / "flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_graph_smoke_v01"
OUT_PKL = OUT_DIR / "R019_stockbridge_swegnn_graph_smoke.pkl"
OUT_JSON = OUT_DIR / "R019_stockbridge_swegnn_graph_smoke.json"

CHANNELS = {
    "H": 0,
    "Vx": 1,
    "Vy": 2,
    "rainfall": 3,
    "DEM": 4,
}

# For the first smoke graph we use grid-cell units.
# Later we can replace this with true raster resolution from the georeferenced DEM.
CELL_SIZE = 1.0


def build_grid_edges(ny: int, nx: int, cell_size: float = 1.0):
    """Create directed 4-neighbour grid edges and geometric edge attributes."""
    node_ids = torch.arange(ny * nx, dtype=torch.long).reshape(ny, nx)

    edges_src = []
    edges_dst = []

    # Horizontal neighbours: left -> right and right -> left
    left = node_ids[:, :-1].reshape(-1)
    right = node_ids[:, 1:].reshape(-1)
    edges_src.extend([left, right])
    edges_dst.extend([right, left])

    # Vertical neighbours: top -> bottom and bottom -> top
    top = node_ids[:-1, :].reshape(-1)
    bottom = node_ids[1:, :].reshape(-1)
    edges_src.extend([top, bottom])
    edges_dst.extend([bottom, top])

    row = torch.cat(edges_src)
    col = torch.cat(edges_dst)

    edge_index = torch.stack([row, col], dim=0)

    # pos = [x, y] in grid units
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

    return edge_index, pos, edge_relative_distance, edge_distance


def main():
    print("=== Prepare Stockbridge SWE-GNN graph smoke ===")
    print("Input:", EVENT_PATH)
    print("Exists:", EVENT_PATH.exists())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tensor = torch.load(EVENT_PATH, map_location="cpu")
    assert tensor.ndim == 4, f"Expected [ny,nx,nt,nc], got {tuple(tensor.shape)}"

    ny, nx, nt, nc = tensor.shape
    num_nodes = ny * nx

    print("\nInput tensor:")
    print("shape:", tuple(tensor.shape))
    print("dtype:", tensor.dtype)
    print("num_nodes:", num_nodes)

    H = tensor[..., CHANNELS["H"]].reshape(num_nodes, nt).contiguous()
    Vx = tensor[..., CHANNELS["Vx"]].reshape(num_nodes, nt).contiguous()
    Vy = tensor[..., CHANNELS["Vy"]].reshape(num_nodes, nt).contiguous()

    rainfall_grid = tensor[..., CHANNELS["rainfall"]].reshape(num_nodes, nt).contiguous()

    # DEM is stored repeated through time in the PT tensor.
    DEM = tensor[:, :, 0, CHANNELS["DEM"]].reshape(num_nodes).contiguous()

    edge_index, pos, edge_relative_distance, edge_distance = build_grid_edges(
        ny=ny,
        nx=nx,
        cell_size=CELL_SIZE,
    )

    row, col = edge_index
    edge_slope = (DEM[col] - DEM[row]) / edge_distance

    # Rainfall appears spatially repeated in these tensors, but we keep both:
    # - rainfall: node-time version for future model forcing
    # - rainfall_global: one representative time series
    rainfall_global = tensor[0, 0, :, CHANNELS["rainfall"]].contiguous()

    qx = Vx * H
    qy = Vy * H
    Qmag = torch.sqrt(qx**2 + qy**2)

    data = Data()
    data.edge_index = edge_index
    data.edge_distance = edge_distance.float()
    data.edge_relative_distance = edge_relative_distance.float()
    data.edge_slope = edge_slope.float()

    data.num_nodes = num_nodes
    data.pos = pos.float()

    data.DEM = DEM.float()
    data.H = H.float()
    data.WD = H.float()  # alias for compatibility with original SWE-GNN naming
    data.Vx = Vx.float()
    data.Vy = Vy.float()
    data.VX = Vx.float()  # alias for original SWE-GNN naming
    data.VY = Vy.float()
    data.Qmag = Qmag.float()

    data.rainfall = rainfall_grid.float()
    data.rainfall_global = rainfall_global.float()

    data.ny = ny
    data.nx = nx
    data.nt = nt
    data.cell_size = CELL_SIZE
    data.event_id = "R019"
    data.split = "test"

    dataset = [data]

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dataset, f)

    metadata = {
        "event_id": "R019",
        "split": "test",
        "input_path": str(EVENT_PATH),
        "output_pkl": str(OUT_PKL),
        "tensor_shape": list(tensor.shape),
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "cell_size": CELL_SIZE,
        "channels": CHANNELS,
        "stored_variables": [
            "edge_index",
            "edge_distance",
            "edge_relative_distance",
            "edge_slope",
            "pos",
            "DEM",
            "H/WD",
            "Vx/VX",
            "Vy/VY",
            "Qmag",
            "rainfall",
            "rainfall_global",
        ],
        "notes": [
            "This is a one-event graph smoke conversion for Stockbridge Phase 2.",
            "Rainfall is stored because Stockbridge events are rainfall-driven.",
            "CELL_SIZE is currently set to 1.0 grid unit for smoke testing.",
        ],
    }

    with open(OUT_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nCreated PyG Data object:")
    print(data)

    print("\nShapes:")
    keys = data.keys() if callable(data.keys) else data.keys
    for key in keys:
        value = getattr(data, key)
        if hasattr(value, "shape"):
            print(f"{key:24s}", tuple(value.shape))
        else:
            print(f"{key:24s}", type(value), value)

    print("\nSaved:")
    print(OUT_PKL)
    print(OUT_JSON)

    print("\nSmoke graph conversion complete.")


if __name__ == "__main__":
    main()
