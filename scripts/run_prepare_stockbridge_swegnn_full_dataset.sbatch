#!/bin/bash
#SBATCH --job-name=swegnn_m13a_dataset
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="$HOME/flood/01_repositories/Stockbridge-SWEGNN"
CONTAINER="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
SCRIPT="$REPO_DIR/scripts/prepare_stockbridge_swegnn_full_dataset.py"

cd "$REPO_DIR"

echo "=== Stockbridge SWE-GNN Milestone 13A full dataset preparation job ==="
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
echo "=== Input dataset quick check ==="
PT_ROOT="$HOME/flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2_core_v01"
for SPLIT in train valid test; do
    echo "$SPLIT count:"
    find "$PT_ROOT/$SPLIT" -maxdepth 1 -name "*.pt" | wc -l
done
echo "R019 in test:"
find "$PT_ROOT/test" -maxdepth 1 -name "*R019*.pt" -print

echo ""
echo "=== Running Milestone 13A dataset preparation ==="
apptainer exec "$CONTAINER" \
    python "$SCRIPT" \
        --cell-size 1.0 \
        --rainfall-scale-reference 300000

echo ""
echo "Milestone 13A full dataset preparation job completed."
