"""M17J-D: Untrained full-domain rollout smoke test.

Tests original and alternative rainfall alignments.
No training, checkpoint writing or dataset modification.
"""

from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

REPOS = Path.home() / "flood/01_repositories"
sys.path.insert(0, str(REPOS / "SWE-GNN-paper-repository"))
sys.path.insert(0, str(REPOS / "Stockbridge-SWEGNN"))

from models.gnn import GNN
from experimental.rainfall_aware_swegnn_v1 import RainfallAwareSWEGNN

DATA_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_100_5m_v01"
)

torch.manual_seed(1884)

assert torch.cuda.is_available(), "This test requires a GPU"
device = torch.device("cuda:0")

static = torch.load(
    DATA_ROOT / "static_graph.pt", map_location="cpu"
)
paths = list(DATA_ROOT.glob("events/*/R006.pt"))
assert len(paths) == 1, paths
event = torch.load(paths[0], map_location="cpu")

n = int(static["num_nodes"])
assert n == 99430
assert int(static["num_edges"]) == 396458

dem = static["DEM"].float().to(device)
sx = static["slope_x"].float().to(device)
sy = static["slope_y"].float().to(device)
edges = static["edge_index"].long().to(device)
edge_attr = static["edge_attr"].float().to(device)

rain = event["rainfall_global"].float().to(device)
rain_extended = torch.cat([rain, rain.new_zeros(1)])

initial_h = event["H"][:, 0].float().to(device)
initial_q = event["Qmag"][:, 0].float().to(device)

assert rain.numel() == 25
assert initial_h.max().item() == 0.0
assert initial_q.max().item() == 0.0

backbone = GNN(
    node_features=7,
    edge_features=3,
    type_GNN="SWEGNN",
    hid_features=8,
    K=2,
    gnn_activation="tanh",
    dropout=0.0,
    mlp_layers=2,
    mlp_activation="prelu",
    seed=1884,
    with_filter_matrix=True,
    with_gradient=True,
    with_WL=True,
    previous_t=1,
    device=device,
).to(device)

model = RainfallAwareSWEGNN(backbone, dem).to(device)
model.eval()

active = model.rainfall_source.active
assert int(active.sum()) == 32249

print("===== M17J-D FULL DOMAIN ROLLOUT =====", flush=True)
print("Device:", torch.cuda.get_device_name(0), flush=True)
print("Nodes:", n, flush=True)
print("Edges:", edges.shape[1], flush=True)
print("Active:", int(active.sum()), flush=True)
print("Weights: RANDOM / UNTRAINED", flush=True)

def run_rollout(alignment):
    h = initial_h.clone()
    q = initial_q.clone()

    print(f"\n===== {alignment.upper()} =====", flush=True)

    with torch.no_grad():
        for t in range(24):
            if alignment == "original":
                rain_a = rain[t]
                rain_b = rain[t + 1]
            elif alignment == "alternative":
                rain_a = rain_extended[t + 1]
                rain_b = rain_extended[t + 2]
            else:
                raise ValueError(alignment)

            x = torch.stack([
                sx,
                sy,
                (rain_a * 300000).expand(n),
                (rain_b * 300000).expand(n),
                dem,
                h,
                q,
            ], dim=1)

            graph = Data(
                x=x,
                edge_index=edges,
                edge_attr=edge_attr,
            )

            prediction = model(graph)
            assert prediction.shape == (n, 2)
            assert torch.isfinite(prediction).all(), (
                f"Non-finite prediction at T{t+1}"
            )

            h = prediction[:, 0]
            q = prediction[:, 1]

            assert (h >= 0).all()
            assert (q >= 0).all()
            assert (h[~active] == 0).all()
            assert (q[~active] == 0).all()

            if t in (0, 1, 3, 7, 11, 23):
                print(
                    f"T{t+1:02d} | "
                    f"rain_next={rain_b.item()*300000:.3f} mm | "
                    f"wet={int((h > 0).sum()):5d} | "
                    f"Hmax={h.max().item():.6f} m | "
                    f"Qmax={q.max().item():.6f}",
                    flush=True,
                )

    return h, q


h_orig, q_orig = run_rollout("original")
h_alt, q_alt = run_rollout("alternative")

print("\n===== FINAL CHECKS =====")
print("24 steps per alignment: PASS")
print("Finite predictions: PASS")
print("Nonnegative predictions: PASS")
print("Inactive cells zero: PASS")
print("Final mean absolute H difference:",
      (h_orig - h_alt).abs().mean().item())
print("Original files unchanged: YES")
print("M17J-D SMOKE COMPLETE", flush=True)
