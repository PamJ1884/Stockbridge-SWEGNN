from pathlib import Path
import torch

PT_ROOT = Path.home() / "flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2_core_v01"
EVENT_PATH = PT_ROOT / "test/R019_stockbridge_T000_T024.pt"

CHANNELS = ["H", "Vx", "Vy", "rainfall", "DEM"]

print("=== Stockbridge Phase 2 PT inspection ===")
print("Path:", EVENT_PATH)
print("Exists:", EVENT_PATH.exists())

tensor = torch.load(EVENT_PATH, map_location="cpu")

print("\nTensor type:", type(tensor))
print("Tensor shape:", tuple(tensor.shape))
print("Tensor dtype:", tensor.dtype)

assert tensor.ndim == 4, f"Expected 4D tensor [H, W, T, C], got {tensor.shape}"

ny, nx, nt, nc = tensor.shape

print("\nDimensions:")
print("ny:", ny)
print("nx:", nx)
print("nt:", nt)
print("nc:", nc)
print("num_nodes:", ny * nx)

print("\nChannel statistics:")
for i, name in enumerate(CHANNELS):
    arr = tensor[..., i]
    print(
        f"{i}: {name:8s} "
        f"shape={tuple(arr.shape)} "
        f"min={arr.min().item():.6f} "
        f"max={arr.max().item():.6f} "
        f"mean={arr.mean().item():.6f}"
    )

print("\nTime checks:")
H = tensor[..., 0]
rain = tensor[..., 3]
DEM = tensor[..., 4]

print("H T0 min/max:", H[:, :, 0].min().item(), H[:, :, 0].max().item())
print("H T24 min/max:", H[:, :, -1].min().item(), H[:, :, -1].max().item())
print("rainfall time series min/max:", rain[0, 0, :].min().item(), rain[0, 0, :].max().item())
print("DEM first time slice min/max:", DEM[:, :, 0].min().item(), DEM[:, :, 0].max().item())

print("\nInspection complete.")
