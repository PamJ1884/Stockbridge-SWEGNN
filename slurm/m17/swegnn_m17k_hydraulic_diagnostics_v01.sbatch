#!/bin/bash
#SBATCH --job-name=m17k_hyd_diag
#SBATCH --account=comet_smhfs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# CPU-only data analysis; submit to an authorized CPU partition with sbatch -p.
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

REPO="$HOME/flood/01_repositories/Stockbridge-SWEGNN"
SCRIPT="$REPO/scripts/stockbridge_m17k_hydraulic_diagnostics_v01.py"
SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"

module load apptainer/1.4.1

test -f "$SCRIPT"
test -f "$SIF"
echo "===== M17K HYDRAULIC DIAGNOSTICS ====="
echo "Date: $(date -Is)"
echo "Host: $(hostname)"
echo "Git revision: $(git -C "$REPO" rev-parse HEAD)"
echo "Script sha256: $(sha256sum "$SCRIPT")"
echo "No test data will be loaded."

apptainer exec "$SIF" python "$SCRIPT" --mode all

echo "===== M17K HYDRAULIC DIAGNOSTICS COMPLETED ====="
