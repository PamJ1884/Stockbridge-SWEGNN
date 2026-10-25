"""Experimental rainfall source head for Stockbridge SWE-GNN.

Not a trained model. Does not modify original SWE-GNN.
The output is an empirical depth increment in metres.
"""

import torch
from torch import nn
from torch.nn import functional as F


class RainfallSourceHead(nn.Module):
    def __init__(self, dem, rain_scale=300000.0, dt=300.0):
        super().__init__()

        dem = dem.flatten().float()
        active = dem != dem.max()

        self.register_buffer("active", active)
        self.register_buffer("dem_mean", dem[active].mean())
        self.register_buffer(
            "dem_std",
            dem[active].std().clamp_min(1.0)
        )

        self.rain_scale = rain_scale
        self.dt = dt

        # Output 0: local rainfall-response threshold (mm)
        # Output 1: nonnegative empirical response multiplier
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x columns: slope_x, slope_y, rain_t,
        # rain_next, DEM, H_t, Qmag_t
        if x.ndim != 2 or x.shape != (self.active.numel(), 7):
            raise ValueError("Unexpected input shape")

        rain_mm = (
            x[:, 3].clamp_min(0)
            * self.dt
            * 1000.0
            / self.rain_scale
        )

        features = torch.stack([
            torch.tanh(x[:, 0]),
            torch.tanh(x[:, 1]),
            (x[:, 4] - self.dem_mean) / self.dem_std,
            rain_mm / 10.0,
        ], dim=1)

        parameters = self.mlp(features)

        threshold_mm = F.softplus(parameters[:, 0])
        multiplier = F.softplus(parameters[:, 1])

        effective_mm = F.relu(rain_mm - threshold_mm)

        return (
            effective_mm
            * multiplier
            / 1000.0
            * self.active.to(x.dtype)
        )
