#!/bin/bash
#SBATCH -J adaptvis.sanity
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/sanity-%j.out
#SBATCH -e logs/sanity-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output outputs

CONDA_BASE="/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate adaptvis_py311

# Use test split for quick validation.
export TEST_MODE=True
export ADAPTVIS_DYNAMIC_PREPROCESS=True

python -u main_aro.py \
  --dataset stress_images_3 \
  --model-name 'llava1.5' \
  --method base \
  --option six

python -u main_aro.py \
  --dataset stress_images_3 \
  --model-name 'llava1.5' \
  --method scaling_vis \
  --weight=0.8 \
  --option six

python -u main_aro.py \
  --dataset stress_images_3 \
  --model-name 'llava1.5' \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.3 \
  --option six
