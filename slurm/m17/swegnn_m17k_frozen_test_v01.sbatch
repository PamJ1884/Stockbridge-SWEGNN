#!/bin/bash
#SBATCH --job-name=m17k_finaltest
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

# M17K frozen evaluation. No fitting, optimizer, or checkpoint selection on test.
# First verify both frozen checkpoints reproduce their saved validation summaries.
# Only after both audits pass, evaluate both models on all 15 test events, T1-T24.
set -euo pipefail
cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"
module load apptainer/1.4.1

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"
ROOT="$HOME/flood/03_outputs/Stockbridge-SWEGNN"
OUT="$ROOT/milestone17k_frozen_final_test_v01"
EVAL="scripts/evaluate_stockbridge_m17k_frozen_v01.py"

for path in "$SIF" "$DATASET/static_graph.pt" "$EVAL" \
  "$ROOT/milestone17k_paperstyle_raw_pilot_v01/rainfall_aware/m17k_best_checkpoint.pt" \
  "$ROOT/milestone17k_k8_raw_pilot_v01/rainfall_aware/m17k_best_checkpoint.pt"; do
  test -f "$path" || { echo "Missing: $path" >&2; exit 1; }
done
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "Refusing to overwrite $OUT" >&2; exit 1
fi
REV="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: Commit changes before frozen evaluation" >&2; exit 1
fi

echo "===== FROZEN M17K VALIDATION AUDITS THEN TEST ====="
echo "Date: $(date); Host: $(hostname); Revision: $REV"
sha256sum "$EVAL" scripts/train_valid_stockbridge_m17k.py experimental/rainfall_aware_swegnn_v2.py
mkdir -p "$OUT"

for model in p4_k4_raw p6b_k8_raw; do
  case "$model" in
    p4_k4_raw) CKPT="$ROOT/milestone17k_paperstyle_raw_pilot_v01/rainfall_aware/m17k_best_checkpoint.pt" ;;
    p6b_k8_raw) CKPT="$ROOT/milestone17k_k8_raw_pilot_v01/rainfall_aware/m17k_best_checkpoint.pt" ;;
  esac
  # Valid only. If either audit fails, 'set -e' prevents all test evaluations.
  apptainer exec --nv "$SIF" python "$EVAL" \
    --dataset-root "$DATASET" --checkpoint "$CKPT" --model "$model" \
    --split valid --output-dir "$OUT/$model/valid_audit"
done

echo "===== BOTH VALIDATION AUDITS PASSED; FROZEN TEST BEGINS ====="
for model in p4_k4_raw p6b_k8_raw; do
  case "$model" in
    p4_k4_raw) CKPT="$ROOT/milestone17k_paperstyle_raw_pilot_v01/rainfall_aware/m17k_best_checkpoint.pt" ;;
    p6b_k8_raw) CKPT="$ROOT/milestone17k_k8_raw_pilot_v01/rainfall_aware/m17k_best_checkpoint.pt" ;;
  esac
  apptainer exec --nv "$SIF" python "$EVAL" \
    --dataset-root "$DATASET" --checkpoint "$CKPT" --model "$model" \
    --split test --audit-json "$OUT/$model/valid_audit/valid_summary.json" \
    --output-dir "$OUT/$model/test"
done

echo "===== M17K FROZEN FINAL TEST COMPLETE ====="
date
