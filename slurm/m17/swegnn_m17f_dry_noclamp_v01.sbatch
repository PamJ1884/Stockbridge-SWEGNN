#!/bin/bash
#SBATCH --job-name=swegnn_m17f_raw
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

set -euo pipefail

cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"

module load apptainer/1.4.1

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"

DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"

CHECKPOINT="$HOME/flood/03_outputs/Stockbridge-SWEGNN/milestone17c_phase2_100_5m_t0_t24_smoke_v01/milestone14b1_best_checkpoint.pt"

OUT="$HOME/flood/03_outputs/Stockbridge-SWEGNN/milestone17f_phase2_100_5m_t0_t24_noclamp_v01"

echo "===== M17F DRY-START NO-CLAMP DIAGNOSTIC ====="
date

test -f "$SIF"
test -f "$CHECKPOINT"
test -f "$DATASET/static_graph.pt"

if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "ERROR: Output directory already exists."
    exit 1
fi

apptainer exec --nv "$SIF" python \
  scripts/evaluate_stockbridge_full_validation_rollout.py \
  --dataset-root "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT" \
  --device cuda \
  --start-time-index 0 \
  --end-time-index 23 \
  --rainfall-scale 300000.0 \
  --hid-features 16 \
  --K 4 \
  --seed 4444 \
  --no-clamp-output \
  --no-clamp-feedback \
  --no-save-predictions

echo "===== M17F COMPLETED ====="
date
