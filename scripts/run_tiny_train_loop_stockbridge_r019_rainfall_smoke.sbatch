#!/bin/bash
#SBATCH --job-name=swegnn_rain_train_r019
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="$HOME/flood/01_repositories/Stockbridge-SWEGNN"
CONTAINER="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
SCRIPT="$REPO_DIR/scripts/tiny_train_loop_stockbridge_r019_rainfall_smoke.py"

cd "$REPO_DIR"

echo "=== Stockbridge SWE-GNN Milestone 7 rainfall tiny training loop smoke job ==="
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
echo "=== Running rainfall tiny training loop: R019, T2->T3 through T16->T17, 2 epochs ==="
apptainer exec --nv "$CONTAINER" \
    python "$SCRIPT" \
        --device auto \
        --start-time-index 2 \
        --end-time-index 16 \
        --epochs 2 \
        --lr 1e-3 \
        --weight-decay 0.0 \
        --rainfall-scale 1.0

echo ""
echo "Milestone 7 rainfall tiny training loop smoke job completed."
