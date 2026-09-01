#!/bin/bash
#SBATCH --job-name=swegnn_m12_rollout
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="$HOME/flood/01_repositories/Stockbridge-SWEGNN"
CONTAINER="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
SCRIPT="$REPO_DIR/scripts/rollout_stockbridge_subset_checkpoint_smoke.py"

cd "$REPO_DIR"

echo "=== Stockbridge SWE-GNN Milestone 12 rollout smoke job ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-NA}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-NA}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NA}"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Python script not found: $SCRIPT"
    exit 1
fi

module load apptainer/1.4.1

echo ""
echo "=== Apptainer ==="
apptainer --version

echo ""
echo "=== GPU ==="
nvidia-smi || true

echo ""
echo "=== Running Milestone 12 rollout smoke ==="
apptainer exec --nv "$CONTAINER" \
    python "$SCRIPT" \
        --device auto \
        --event-id R002 \
        --start-time-index 2 \
        --end-time-index 16 \
        --rainfall-scale 300000 \
        --clamp-nonnegative \
        --save-predictions \
        --overwrite

echo ""
echo "Milestone 12 rollout smoke job completed."
