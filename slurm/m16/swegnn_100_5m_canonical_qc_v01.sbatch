#!/bin/bash
#SBATCH --job-name=swegnn100_canonical_qc
#SBATCH --account=comet_smhfs
#SBATCH --partition=default_free
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out

set -euo pipefail

cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"

module load apptainer/1.4.1

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"

echo "===== SWE-GNN CANONICAL QC ====="
date

apptainer exec "$SIF" python \
    scripts/qc_stockbridge_swegnn_100_5m_v01.py

echo "===== QC COMPLETED ====="
date
