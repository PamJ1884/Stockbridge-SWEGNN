#!/bin/bash
#SBATCH --job-name=m17k_paperstyle
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

# M17K-P: original SWE-GNN training strategy, full 70/15 data, from scratch.
#
# Reproduces Bentivoglio et al. (2023) Sect. 3.3 / Eq. 14 / Algorithm 1 as
# implemented in SWE-GNN-paper-repository (config.yaml, training/):
#   loss         RMSE per variable over wet cells (pred or target nonzero),
#                weights 1 (H) and 3 (Qmag), mean over rollout steps
#   curriculum   rollout steps = min(epoch // 15 + 1, 8), up to 150 epochs
#   optimiser    Adam, lr 0.007, StepLR(step 7, gamma 0.9), batch 8
#                (here: gradient accumulation over 8 samples)
#   clipping     element-wise gradient clipping at 0.5
#   selection    validation rollout RMSE where there is water
#   stopping     30 consecutive epochs without improving on the previous one
#
# Declared deviations (unchanged from earlier M17K runs): hid_features 16 and
# K 4 (paper config: 64 and 8), previous_t 1 (paper config: 2), loss restricted
# to active hydraulic cells, alternative rainfall alignment, and the
# rainfall-aware v2 source for ARCH=rainfall_aware.
#
# Usage:
#   sbatch --export=ALL,ARCH=rainfall_aware slurm/m17/swegnn_m17k_paperstyle_v01.sbatch
#
# Cost: every start time of every event is used, as in the original.
# Estimated 8-10 h on an L40S from the one-step pilot timing (not yet measured).
# If the partition time limit is shorter, add e.g. --max-starts-per-event 8.

set -euo pipefail

cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"

module load apptainer/1.4.1

ARCH="${ARCH:?Set ARCH=swegnn or rainfall_aware}"

case "$ARCH" in
    swegnn)
        ARCH_ARGS=(--architecture swegnn)
        ;;
    rainfall_aware)
        ARCH_ARGS=(
            --architecture rainfall_aware
            --rain-source-input trapezoid
            --no-backbone-depth-mask
        )
        ;;
    *)
        echo "ERROR: Unknown architecture $ARCH"
        exit 1
        ;;
esac

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"
OUT="$HOME/flood/03_outputs/Stockbridge-SWEGNN/milestone17k_paperstyle_v01/$ARCH"

test -f "$SIF"
test -f "$DATASET/static_graph.pt"

if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "ERROR: Refusing to overwrite $OUT"
    exit 1
fi

REVISION="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
    REVISION="${REVISION}-dirty"
fi

echo "===== M17K-P PAPER-STYLE TRAINING ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Revision: $REVISION"
echo "Architecture: $ARCH"
echo "Output: $OUT"

sha256sum \
    scripts/train_valid_stockbridge_m17k.py \
    experimental/rainfall_aware_swegnn_v2.py

apptainer exec --nv --env "M17K_GIT_REVISION=$REVISION" "$SIF" python \
    scripts/train_valid_stockbridge_m17k.py \
    --dataset-root "$DATASET" \
    --output-dir "$OUT" \
    --device cuda \
    --rainfall-alignment alternative \
    --start-time-index 0 \
    --end-time-index 23 \
    --loss swegnn_rmse \
    --loss-q-weight 3 \
    --loss-mask where_water \
    --curriculum-epochs 15 \
    --max-rollout-steps 8 \
    --epochs 150 \
    --lr 0.007 \
    --lr-step-size 7 \
    --lr-gamma 0.9 \
    --accumulate-samples 8 \
    --grad-clip-value 0.5 \
    --grad-clip-norm 0 \
    --select-metric rollout_swegnn_loss \
    --early-stopping consecutive_increase \
    --patience 30 \
    --rainfall-scale 300000.0 \
    --hid-features 16 \
    --K 4 \
    --seed 4444 \
    "${ARCH_ARGS[@]}"

echo "===== M17K-P COMPLETED ====="
date
