#!/bin/bash
#SBATCH --job-name=m17jd_r006
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

module load apptainer/1.4.1

ROOT="$HOME/flood/01_repositories/Stockbridge-SWEGNN"
SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"

export OMP_NUM_THREADS=4

cd "$ROOT"

apptainer exec --nv "$SIF" python \
  experimental/smoke_r006_rollout_v1.py
