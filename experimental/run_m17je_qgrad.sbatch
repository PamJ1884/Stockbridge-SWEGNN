#!/bin/bash
#SBATCH --job-name=m17je_qgrad
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail

module load apptainer/1.4.1

cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"

apptainer exec --nv "$SIF" python \
  experimental/diagnose_m17je_qgrad.py
