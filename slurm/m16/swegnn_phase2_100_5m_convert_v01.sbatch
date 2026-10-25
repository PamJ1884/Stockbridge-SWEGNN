#!/bin/bash
#SBATCH --job-name=swegnn100_5m_convert
#SBATCH --account=comet_smhfs
#SBATCH --partition=default_free
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out

set -euo pipefail

cd "$HOME/flood/01_repositories/Stockbridge-SWEGNN"

module load apptainer/1.4.1

SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"

SRC="$HOME/flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2_core_v01"

OUT="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"

echo "===== SWE-GNN 100 CORRECTED CONVERSION ====="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Source: $SRC"
echo "Output: $OUT"
echo "Cell size: 5.0 m"

test -f "$SIF"
test -d "$SRC"

if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "ERROR: Output already exists. Refusing to overwrite."
    exit 1
fi

apptainer exec "$SIF" python \
    scripts/prepare_stockbridge_swegnn_full_dataset.py \
    --pt-root "$SRC" \
    --output-root "$OUT" \
    --cell-size 5.0 \
    --rainfall-scale-reference 300000.0

echo "===== CONVERSION COMPLETED ====="
echo "Date: $(date)"

du -sh "$OUT"
