#!/bin/bash
#SBATCH --job-name=swegnn_m15a_r019
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="$HOME/flood/01_repositories/Stockbridge-SWEGNN"
CONTAINER="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
SCRIPT="$REPO_DIR/scripts/evaluate_stockbridge_swegnn_test_r019.py"

cd "$REPO_DIR"

echo "=== Stockbridge SWE-GNN Milestone 15A test/R019 evaluation job ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-NA}"
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
echo "=== Running Milestone 15A SWE-GNN test/R019 evaluation ==="
apptainer exec --nv "$CONTAINER" \
    python "$SCRIPT" \
        --device auto \
        --start-time-index 2 \
        --end-time-index 16 \
        --rainfall-scale 300000 \
        --hid-features 16 \
        --K 4 \
        --seed 4444 \
        --modes teacher_forced,autoregressive \
        --thresholds 0.05,0.10,0.20 \
        --clamp-output \
        --clamp-feedback \
        --dpi 180 \
        --max-depth-vmax 0 \
        --error-vmax 0 \
        --overwrite

echo ""
echo "Milestone 15A SWE-GNN test/R019 evaluation job completed."
