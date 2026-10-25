#!/bin/bash
#SBATCH --job-name=swegnn_m17c_smoke
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

set -euo pipefail

cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"

module load apptainer/1.4.1

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"

DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"

OUT="$HOME/flood/03_outputs/Stockbridge-SWEGNN/milestone17c_phase2_100_5m_t0_t24_smoke_v01"

echo "===== M17C SWE-GNN FULL-HORIZON GPU SMOKE ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Dataset: $DATASET"
echo "Output: $OUT"
echo "Temporal range: T0 -> T24"
echo "Train events: 1"
echo "Validation events: 1"

test -f "$SIF"
test -f "$DATASET/static_graph.pt"

if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "ERROR: Output already exists. Refusing overwrite."
    exit 1
fi

echo "===== GPU CHECK ====="

apptainer exec --nv "$SIF" python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA GPU unavailable"

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== TRAINING SMOKE ====="

apptainer exec --nv "$SIF" python \
    scripts/train_valid_stockbridge_full_onestep_earlystop.py \
    --dataset-root "$DATASET" \
    --output-dir "$OUT" \
    --device cuda \
    --start-time-index 0 \
    --end-time-index 23 \
    --epochs 1 \
    --patience 1 \
    --max-train-events 1 \
    --max-valid-events 1 \
    --rainfall-scale 300000.0 \
    --hid-features 16 \
    --K 4 \
    --seed 4444

echo "===== M17C GPU SMOKE COMPLETED ====="
echo "Date: $(date)"
du -sh "$OUT"
