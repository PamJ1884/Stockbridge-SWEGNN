#!/bin/bash
#SBATCH --job-name=swegnn_m17k
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

set -euo pipefail

cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"
module load apptainer/1.4.1

ARCH="${ARCH:?Set ARCH=swegnn or rainfall_aware}"
STAGE="${STAGE:?Set STAGE=preflight or smoke}"

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

case "$STAGE" in
    preflight)
        EPOCHS=1
        TRAIN_EVENTS=1
        VALID_EVENTS=1
        ;;
    smoke)
        EPOCHS=5
        TRAIN_EVENTS=5
        VALID_EVENTS=3
        ;;
    pilot)
        EPOCHS=3
        TRAIN_EVENTS=0
        VALID_EVENTS=0
        ;;
    *)
        echo "ERROR: Unknown stage $STAGE"
        exit 1
        ;;
esac

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"
OUT="$HOME/flood/03_outputs/Stockbridge-SWEGNN/milestone17k_${STAGE}_v01/$ARCH"

test -f "$SIF"
test -f "$DATASET/static_graph.pt"

if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "ERROR: Refusing to overwrite $OUT"
    exit 1
fi

echo "===== M17K TRAINING ====="
echo "Job: $SLURM_JOB_ID"
echo "Architecture: $ARCH"
echo "Stage: $STAGE"
echo "Git: $(git rev-parse HEAD)"
echo "Date: $(date)"

REVISION="$(git rev-parse HEAD)"

if [ -n "$(git status --porcelain)" ]; then
    REVISION="${REVISION}-dirty"
fi

echo "M17K Git revision: $REVISION"

echo "===== SOURCE FILE CHECKSUMS ====="
sha256sum \
    scripts/train_valid_stockbridge_m17k.py \
    experimental/rainfall_aware_swegnn_v2.py

apptainer exec --nv --env "M17K_GIT_REVISION=$REVISION" "$SIF" python \
    scripts/train_valid_stockbridge_m17k.py \
    --dataset-root "$DATASET" \
    --output-dir "$OUT" \
    --device cuda \
    --rainfall-alignment alternative \
    --loss-mask active \
    --start-time-index 0 \
    --end-time-index 23 \
    --epochs "$EPOCHS" \
    --patience "$EPOCHS" \
    --max-train-events "$TRAIN_EVENTS" \
    --max-valid-events "$VALID_EVENTS" \
    --rainfall-scale 300000.0 \
    --hid-features 16 \
    --K 4 \
    --seed 4444 \
    "${ARCH_ARGS[@]}"

echo "===== M17K COMPLETED ====="
date
