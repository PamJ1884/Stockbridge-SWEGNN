#!/bin/bash
#SBATCH --job-name=swegnn_m17k_smoke
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

# M17K supervised smoke: same small budget for
#   A = original SWE-GNN       (backbone 1 mm mask on, no rainfall source)
#   C = rainfall-aware SWE-GNN (v2: trapezoid source, backbone mask off)
# Both: alternative rainfall alignment, loss on active cells, T0 -> T24,
# 5 train events, 3 valid events, 5 epochs, then autoregressive rollout
# from T0 on the validation events with the best checkpoint.
# This is a pipeline test, not a result.

set -euo pipefail

cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"

module load apptainer/1.4.1

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"
OUT_ROOT="$HOME/flood/03_outputs/Stockbridge-SWEGNN/milestone17k_smoke_v01"

echo "===== M17K SWE-GNN SUPERVISED SMOKE ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
git status --short
echo "Dataset: $DATASET"
echo "Output root: $OUT_ROOT"

test -f "$SIF"
test -f "$DATASET/static_graph.pt"

if [ -e "$OUT_ROOT" ] || [ -L "$OUT_ROOT" ]; then
    echo "ERROR: Output already exists. Refusing overwrite."
    exit 1
fi

COMMON_ARGS=(
    --dataset-root "$DATASET"
    --device cuda
    --rainfall-alignment alternative
    --loss-mask active
    --start-time-index 0
    --end-time-index 23
    --epochs 5
    --patience 5
    --max-train-events 5
    --max-valid-events 3
    --rainfall-scale 300000.0
    --hid-features 16
    --K 4
    --seed 4444
)

echo "===== A: ORIGINAL SWE-GNN ====="
apptainer exec --nv "$SIF" python \
    scripts/train_valid_stockbridge_m17k.py \
    "${COMMON_ARGS[@]}" \
    --architecture swegnn \
    --output-dir "$OUT_ROOT/A_swegnn"

echo "===== C: RAINFALL-AWARE SWE-GNN v2 ====="
apptainer exec --nv "$SIF" python \
    scripts/train_valid_stockbridge_m17k.py \
    "${COMMON_ARGS[@]}" \
    --architecture rainfall_aware \
    --rain-source-input trapezoid \
    --no-backbone-depth-mask \
    --output-dir "$OUT_ROOT/C_rainfall_aware"

echo "===== M17K SMOKE COMPLETED ====="
echo "Date: $(date)"
du -sh "$OUT_ROOT"
