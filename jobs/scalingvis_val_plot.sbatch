#!/bin/bash
#SBATCH -J scalingvis.plot
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -p cpu
#SBATCH -o logs/scalingvis-plot-%j.out
#SBATCH -e logs/scalingvis-plot-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs sweeps/scalingvis/reports

CONDA_BASE="/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate adaptvis_py311

python scripts/scalingvis_val_report.py --root sweeps/scalingvis
