#!/bin/bash
#SBATCH --job-name=swegnn_m14b5_ext
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
SCRIPT="$REPO_DIR/scripts/plot_stockbridge_validation_extent_confusion.py"

cd "$REPO_DIR"

echo "=== Stockbridge SWE-GNN Milestone 14B-5 validation extent confusion job ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-NA}"

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
echo "=== Running Milestone 14B-5 validation extent confusion diagnostic ==="
apptainer exec "$CONTAINER" \
    python "$SCRIPT" \
        --event-ids auto \
        --modes teacher_forced,autoregressive \
        --thresholds 0.05,0.10,0.20 \
        --stages maximum,final \
        --dpi 180 \
        --overwrite

echo ""
echo "Milestone 14B-5 validation extent confusion job completed."
