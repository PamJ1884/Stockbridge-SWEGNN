#!/bin/bash
#SBATCH --job-name=m17kp_preflight
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

# M17K-P GPU preflight. One train event and one validation event.
# P1 = 2-step BPTT; P2 = 8-step BPTT; both use 8 sampled starts/event.
# Do NOT use these debugging-subset checkpoints for a final model comparison.
# Usage: sbatch --export=ALL,STAGE=P1 slurm/m17/swegnn_m17k_paperstyle_preflight_v01.sbatch
#        sbatch --export=ALL,STAGE=P2 slurm/m17/swegnn_m17k_paperstyle_preflight_v01.sbatch

set -euo pipefail
cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"
module load apptainer/1.4.1

STAGE="${STAGE:?Specify STAGE=P1 or STAGE=P2}"
case "$STAGE" in
  P1) ROLLOUT_SCHEDULE="2:1" ;;
  P2) ROLLOUT_SCHEDULE="8:1" ;;
  *) echo "ERROR: Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"
OUT="$HOME/flood/03_outputs/Stockbridge-SWEGNN/milestone17k_paperstyle_preflight_v01/$STAGE"

test -f "$SIF"
test -f "$DATASET/static_graph.pt"
test -f scripts/train_valid_stockbridge_m17k.py
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "ERROR: Refusing to overwrite $OUT" >&2
  exit 1
fi

REVISION="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  REVISION="${REVISION}-dirty"
fi

echo "===== M17K-P GPU PREFLIGHT ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Git revision: $REVISION"
echo "Stage: $STAGE; rollout schedule: $ROLLOUT_SCHEDULE"
echo "Train/valid limits: 1/1; starts/event: 8; gradient accumulation: 8"
echo "Out: $OUT"
sha256sum scripts/train_valid_stockbridge_m17k.py experimental/rainfall_aware_swegnn_v2.py

apptainer exec --nv --env "M17K_GIT_REVISION=$REVISION" "$SIF" python \
  scripts/train_valid_stockbridge_m17k.py \
  --dataset-root "$DATASET" \
  --output-dir "$OUT" \
  --device cuda \
  --architecture rainfall_aware \
  --rain-source-input trapezoid \
  --no-backbone-depth-mask \
  --rainfall-alignment alternative \
  --start-time-index 0 \
  --end-time-index 23 \
  --max-train-events 1 \
  --max-valid-events 1 \
  --rollout-schedule "$ROLLOUT_SCHEDULE" \
  --epochs 1 \
  --max-starts-per-event 8 \
  --loss swegnn_rmse \
  --loss-q-weight 3 \
  --loss-mask where_water \
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
  --seed 4444

echo "===== M17K-P GPU PREFLIGHT $STAGE COMPLETED ====="
date
