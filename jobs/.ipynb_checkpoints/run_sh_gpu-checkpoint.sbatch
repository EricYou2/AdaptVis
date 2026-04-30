#!/bin/bash
#SBATCH -J adaptvis.runsh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/runsh-%j.out
#SBATCH -e logs/runsh-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output outputs

CONDA_BASE="/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate adaptvis_py311

bash run.sh
