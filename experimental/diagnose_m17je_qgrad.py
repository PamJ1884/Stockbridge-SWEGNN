"""M17J-E: Wet-start Qmag and gradient diagnostic.

Uses R006 T8 -> T9, alternative rainfall alignment.
No optimizer step, no checkpoint and no dataset writes.
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

ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/swegnn"
    / "stockbridge_phase2_100_5m_v01"
)

torch.manual_seed(1884)
device = torch.device("cuda:0")
assert torch.cuda.is_available()

static = torch.load(ROOT / "static_graph.pt", map_location="cpu")
paths = list(ROOT.glob("events/*/R006.pt"))
assert len(paths) == 1
event = torch.load(paths[0], map_location="cpu")

def gpu(tensor):
    return tensor.float().to(device)

dem = gpu(static["DEM"])
sx = gpu(static["slope_x"])
sy = gpu(static["slope_y"])
rain = gpu(event["rainfall_global"])
edge_index = static["edge_index"].long().to(device)
edge_attr = gpu(static["edge_attr"])

t = 8
n = dem.numel()

# Alternative temporal alignment:
# PT r[t+1] and r[t+2] correspond to
# original TXT times t*300 and (t+1)*300 seconds.
x = torch.stack([
    sx,
    sy,
    (rain[t+1] * 300000).expand(n),
    (rain[t+2] * 300000).expand(n),
    dem,
    gpu(event["H"])[:, t],
    gpu(event["Qmag"])[:, t],
], dim=1)

target = torch.stack([
    gpu(event["H"])[:, t+1],
    gpu(event["Qmag"])[:, t+1],
], dim=1)

graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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
active = model.rainfall_source.active

print("===== M17J-E WET-START DIAGNOSTIC =====", flush=True)
print("Event: R006 | T8 -> T9", flush=True)
print("Weights: RANDOM / UNTRAINED", flush=True)

with torch.no_grad():
    raw = model.backbone(graph)
    predicted = model(graph)
    source = model.rainfall_source(x)

    print("True H(T8) max:", x[:, 5].max().item())
    print("True Qmag(T8) max:", x[:, 6].max().item())
    print("True Qmag(T9) max:", target[:, 1].max().item())

    print("Raw backbone Qmag min:", raw[:, 1].min().item())
    print("Raw backbone Qmag max:", raw[:, 1].max().item())
    print("Raw backbone Qmag positive cells:",
          int((raw[:, 1] > 0).sum()))

    print("Wrapped Qmag positive cells:",
          int((predicted[:, 1] > 0).sum()))
    print("Wrapped Qmag max:", predicted[:, 1].max().item())

    print("Rainfall source positive cells:",
          int((source > 0).sum()))

# One supervised backward pass. No optimizer step.
model.train()
model.zero_grad(set_to_none=True)

prediction = model(graph)

loss_h = (
    (prediction[active, 0] - target[active, 0]) ** 2
).mean()

loss_q = (
    (prediction[active, 1] - target[active, 1]) ** 2
).mean()

total_loss = loss_h + loss_q

assert torch.isfinite(total_loss)

backbone_params = tuple(model.backbone.parameters())

q_grads = torch.autograd.grad(
    loss_q,
    backbone_params,
    retain_graph=True,
    allow_unused=True,
)

q_grad_max = max(
    (g.abs().max().item() for g in q_grads if g is not None),
    default=0.0,
)

del q_grads

total_loss.backward()

def max_gradient(module):
    return max(
        (
            p.grad.abs().max().item()
            for p in module.parameters()
            if p.grad is not None
        ),
        default=0.0,
    )

backbone_grad = max_gradient(model.backbone)
source_grad = max_gradient(model.rainfall_source)

print("\n===== GRADIENT CHECK =====")
print("H loss:", loss_h.item())
print("Qmag loss:", loss_q.item())
print("Qmag-only backbone gradient max:", q_grad_max)
print("Combined backbone gradient max:", backbone_grad)
print("Rainfall source gradient max:", source_grad)

assert torch.isfinite(
    torch.tensor([q_grad_max, backbone_grad, source_grad])
).all()

print("Finite gradients: PASS")
print("Backbone receives gradients:", backbone_grad > 0)
print("Rainfall source receives gradients:", source_grad > 0)
print("Qmag loss reaches backbone:", q_grad_max > 0)
print("No model weights updated.")
print("M17J-E DIAGNOSTIC COMPLETE", flush=True)
