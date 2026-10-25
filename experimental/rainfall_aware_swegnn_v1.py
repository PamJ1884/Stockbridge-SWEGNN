"""Experimental rainfall-aware wrapper for original SWE-GNN.

The original backbone remains unchanged.
This is a research prototype, not a mass-conserving solver.
"""

import torch
from torch import nn

from experimental.rainfall_source_v1 import RainfallSourceHead


class RainfallAwareSWEGNN(nn.Module):
    def __init__(self, backbone: nn.Module, dem: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.rainfall_source = RainfallSourceHead(dem)

    def forward(self, graph):
        baseline = self.backbone(graph)

        if baseline.ndim != 2 or baseline.shape[1] != 2:
            raise ValueError(
                f"Expected SWE-GNN output [N, 2], "
                f"got {tuple(baseline.shape)}"
            )

        rainfall_increment = self.rainfall_source(graph.x)

        if rainfall_increment.shape[0] != baseline.shape[0]:
            raise ValueError("Rainfall source and backbone size mismatch")

        active = self.rainfall_source.active

        # Hydraulic prediction + learned empirical rainfall contribution.
        h = (baseline[:, 0] + rainfall_increment).clamp_min(0.0)
        h = torch.where(active, h, torch.zeros_like(h))

        # Qmag is a nonnegative magnitude.
        q = baseline[:, 1].clamp_min(0.0)
        q = torch.where(
            active & (h > 0),
            q,
            torch.zeros_like(q)
        )

        return torch.stack([h, q], dim=1)
