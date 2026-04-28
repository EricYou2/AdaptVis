#!/bin/bash
#SBATCH -J adaptvis.ablationA
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/ablationA-%j.out
#SBATCH -e logs/ablationA-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output outputs

CONDA_BASE="/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate adaptvis_py311

# Keep the repo's held-out complement split for evaluation.
export TEST_MODE=True
# Compatibility mode for the fixed-length preprocessing path on Oscar.
export ADAPTVIS_DYNAMIC_PREPROCESS=True
export PYTHONNOUSERSITE=True

# WhatsUp in this repo is represented by the VG-QA two-object benchmark split.
DATASET="${DATASET:-VG_QA_two_obj}"
MODEL_NAME="${MODEL_NAME:-llava1.5}"
METHOD="${METHOD:-adapt_vis}"
OPTION="${OPTION:-six}"
WEIGHT="${WEIGHT:-1.0}"
WEIGHT1="${WEIGHT1:-1.0}"   # no smoothing
WEIGHT2="${WEIGHT2:-2.0}"   # max sharpening from the README's suggested range
THRESHOLD="${THRESHOLD:-0.4}"
SEED="${SEED:-1}"
DOWNLOAD_FLAG="${DOWNLOAD_FLAG:---download}"

echo "=== Ablation A: Sharpening Only ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Dataset: $DATASET"
echo "Option count: $OPTION"
echo "Weight1: $WEIGHT1"
echo "Weight2: $WEIGHT2"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
nvidia-smi || true

set -x
python -u main_aro.py \
  --dataset "$DATASET" \
  --model-name "$MODEL_NAME" \
  --method "$METHOD" \
  --option "$OPTION" \
  --weight "$WEIGHT" \
  --weight1 "$WEIGHT1" \
  --weight2 "$WEIGHT2" \
  --threshold "$THRESHOLD" \
  --seed "$SEED" \
  --num_workers 1 \
  $DOWNLOAD_FLAG
