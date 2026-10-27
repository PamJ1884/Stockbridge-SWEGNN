#!/bin/bash
#SBATCH --job-name=m17kp_k8raw_p6
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

# M17K-P6b: K=8 RAW full-data pilot.
# Controlled comparison against M17K-P4 RAW (K=4).
#
# Train: 70 events. Validation: 15 events. Test: not used.
# Four epochs with rollout schedule 1, 2, 4, 8.
# All available training starts per event.
# Same architecture, loss, optimiser, seed and selection as P4.
# Only model hyperparameter changed: K=4 -> K=8.
#
# Output is isolated to avoid overwriting P4.
# This pilot is not the final 150-epoch experiment.

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
OUT="$HOME/flood/03_outputs/Stockbridge-SWEGNN/milestone17k_k8_raw_pilot_v01/$ARCH"

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
    --rollout-schedule '1:1,2:1,4:1,8:1' \
    --max-rollout-steps 8 \
    --epochs 4 \
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
    --K 8 \
    --seed 4444 \
    --output-constraint raw \
    "${ARCH_ARGS[@]}"

echo "===== M17K-P COMPLETED ====="
date
