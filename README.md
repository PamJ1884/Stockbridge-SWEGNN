# Stockbridge-SWEGNN

Stockbridge-specific SWE-GNN adaptation workspace for the Research Pam project.

This workspace uses the original SWE-GNN repository as reference, but keeps the Stockbridge Phase 2 adaptation separate.

Context:
- Dataset: Stockbridge Phase 2 CityCAT dataset
- Simulations: 100
- Split: 70 train / 15 valid / 15 test
- Tensor shape: [305, 326, 25, 5]
- Channels: H, Vx, Vy, rainfall, DEM
- Forecast horizon: T0 -> T1-T24
- Time step: 5 minutes
- Visual comparison event: R019

Container:
~/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif
