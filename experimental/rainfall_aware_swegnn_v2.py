"""M17K: rainfall-aware SWE-GNN, version 2.

Changes with respect to v1 (v1 is kept unchanged for reproducibility):

1. The rainfall source can use the exact interval depth. CityCAT interpolates
   rainfall intensity linearly between TXT records (verified with the closed
   flat box test, 2026-10-25), so the depth that falls between two records is
   the trapezoid of the two intensities. With the alternative alignment the two
   rainfall features are exactly the start and end of the predicted interval.
2. The backbone's internal 1 mm depth mask can be disabled. The original
   SWE-GNN zeroes any depth below 1 mm (and the velocity there) before the
   rainfall source is added, which prevents thin rainfall sheets from
   accumulating. With the mask disabled only the final output constraints of
   this wrapper apply.
3. The hydraulic mask is passed in explicitly instead of being rebuilt
   inside the source head.
4. Output constraints are configurable (output_constraint):
   - "clamp" (v1 behaviour): H and Qmag are clamped to >= 0 and Qmag is zeroed
     where H is zero. Clamped cells receive exactly zero gradient, so a cell
     whose raw Qmag turns negative cannot recover during training.
   - "raw": the unclamped prediction (zero only outside the hydraulic mask) is
     returned, as in the original SWE-GNN, whose loss uses the raw output.
     The caller clamps the state that is fed back and the metrics.
   The last unclamped output is kept in `last_raw` for diagnostics.

This is still an empirical research prototype, not a mass-conserving solver.
The original SWE-GNN code is not modified: the depth mask is switched off on
the backbone instance only.
"""

import torch
from torch import nn
from torch.nn import functional as F

RAIN_T_COLUMN = 2
RAIN_NEXT_COLUMN = 3
RAIN_SOURCE_INPUTS = ("next", "trapezoid")


def active_mask_from_dem(dem: torch.Tensor) -> torch.Tensor:
    """Stockbridge convention: inactive cells carry the artificial maximum DEM."""
    dem = dem.flatten()
    return dem != dem.max()


def _identity_depth_mask(x, epsilon=0.001):
    return x


class RainfallSourceHeadV2(nn.Module):
    def __init__(
        self,
        dem: torch.Tensor,
        active: torch.Tensor,
        rain_source_input: str = "trapezoid",
        rain_scale: float = 300000.0,
        dt: float = 300.0,
    ):
        super().__init__()
        if rain_source_input not in RAIN_SOURCE_INPUTS:
            raise ValueError(
                f"rain_source_input must be one of {RAIN_SOURCE_INPUTS}, "
                f"got {rain_source_input!r}"
            )
        dem = dem.flatten().float()
        active = active.flatten().bool()
        if dem.shape != active.shape:
            raise ValueError("DEM and active mask must have the same size")
        if not active.any():
            raise ValueError("Active mask selects no cells")

        self.register_buffer("active", active)
        self.register_buffer("dem_mean", dem[active].mean())
        self.register_buffer("dem_std", dem[active].std().clamp_min(1.0))
        self.rain_source_input = rain_source_input
        self.rain_scale = float(rain_scale)
        self.dt = float(dt)

        # Output 0: local rainfall-response threshold (mm)
        # Output 1: nonnegative empirical response multiplier
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def interval_rain_mm(self, x: torch.Tensor) -> torch.Tensor:
        """Rainfall depth (mm) of the predicted interval, per node."""
        to_mm = self.dt * 1000.0 / self.rain_scale
        rain_next = x[:, RAIN_NEXT_COLUMN].clamp_min(0) * to_mm
        if self.rain_source_input == "next":
            return rain_next
        rain_t = x[:, RAIN_T_COLUMN].clamp_min(0) * to_mm
        return 0.5 * (rain_t + rain_next)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x columns: slope_x, slope_y, rain_t, rain_next, DEM, H_t, Qmag_t
        if x.ndim != 2 or x.shape != (self.active.numel(), 7):
            raise ValueError(f"Unexpected input shape {tuple(x.shape)}")

        rain_mm = self.interval_rain_mm(x)
        features = torch.stack(
            [
                torch.tanh(x[:, 0]),
                torch.tanh(x[:, 1]),
                (x[:, 4] - self.dem_mean) / self.dem_std,
                rain_mm / 10.0,
            ],
            dim=1,
        )
        parameters = self.mlp(features)
        threshold_mm = F.softplus(parameters[:, 0])
        multiplier = F.softplus(parameters[:, 1])
        effective_mm = F.relu(rain_mm - threshold_mm)
        return effective_mm * multiplier / 1000.0 * self.active.to(x.dtype)


class RainfallAwareSWEGNNV2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dem: torch.Tensor,
        active: torch.Tensor,
        rain_source_input: str = "trapezoid",
        backbone_depth_mask: bool = False,
        rain_scale: float = 300000.0,
        dt: float = 300.0,
        output_constraint: str = "clamp",
    ):
        super().__init__()
        if output_constraint not in ("clamp", "raw"):
            raise ValueError(
                f"output_constraint must be 'clamp' or 'raw', got {output_constraint!r}"
            )
        self.output_constraint = output_constraint
        self.last_raw = None
        self.backbone = backbone
        self.backbone_depth_mask = bool(backbone_depth_mask)
        if not self.backbone_depth_mask:
            if not hasattr(backbone, "_mask_small_WD"):
                raise AttributeError("Backbone has no _mask_small_WD to disable")
            backbone._mask_small_WD = _identity_depth_mask
        self.rainfall_source = RainfallSourceHeadV2(
            dem,
            active,
            rain_source_input=rain_source_input,
            rain_scale=rain_scale,
            dt=dt,
        )

    @property
    def active(self) -> torch.Tensor:
        return self.rainfall_source.active

    def forward(self, graph):
        baseline = self.backbone(graph)
        if baseline.ndim != 2 or baseline.shape[1] != 2:
            raise ValueError(
                f"Expected SWE-GNN output [N, 2], got {tuple(baseline.shape)}"
            )
        rainfall_increment = self.rainfall_source(graph.x)
        if rainfall_increment.shape[0] != baseline.shape[0]:
            raise ValueError("Rainfall source and backbone size mismatch")

        active = self.active
        h_raw = torch.where(
            active, baseline[:, 0] + rainfall_increment, torch.zeros_like(baseline[:, 0])
        )
        q_raw = torch.where(active, baseline[:, 1], torch.zeros_like(baseline[:, 1]))
        self.last_raw = torch.stack([h_raw, q_raw], dim=1).detach()
        if self.output_constraint == "raw":
            return torch.stack([h_raw, q_raw], dim=1)

        h = h_raw.clamp_min(0.0)
        q = q_raw.clamp_min(0.0)
        q = torch.where(h > 0, q, torch.zeros_like(q))
        return torch.stack([h, q], dim=1)
