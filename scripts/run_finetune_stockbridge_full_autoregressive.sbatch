#!/bin/bash
#SBATCH --job-name=swegnn_m14b3_arft
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="$HOME/flood/01_repositories/Stockbridge-SWEGNN"
CONTAINER="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
SCRIPT="$REPO_DIR/scripts/finetune_stockbridge_full_autoregressive.py"

cd "$REPO_DIR"

echo "=== Stockbridge SWE-GNN Milestone 14B-3 autoregressive fine-tuning job ==="
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
echo "=== Running Milestone 14B-3 autoregressive fine-tuning ==="
apptainer exec --nv "$CONTAINER" \
    python "$SCRIPT" \
        --device auto \
        --start-time-index 2 \
        --end-time-index 16 \
        --unroll-steps 5 \
        --epochs 30 \
        --patience 8 \
        --min-delta 1e-8 \
        --lr 5e-4 \
        --weight-decay 0.0 \
        --grad-clip-norm 1.0 \
        --rainfall-scale 300000 \
        --hid-features 16 \
        --K 4 \
        --seed 4444 \
        --shuffle \
        --clamp-output \
        --clamp-feedback \
        --no-save-predictions \
        --overwrite

echo ""
echo "Milestone 14B-3 autoregressive fine-tuning job completed."
