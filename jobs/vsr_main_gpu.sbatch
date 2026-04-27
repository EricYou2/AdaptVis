#!/bin/bash
#SBATCH -J adaptvis.vsr
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/vsr_main-%j.out
#SBATCH -e logs/vsr_main-%j.err

set -euo pipefail

# Run from the directory where sbatch is invoked.
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs output outputs

# Activate your conda environment on Oscar.
CONDA_BASE="/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  # Fallback for future path changes.
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate adaptvis_py311

# Optional split control: True uses the 80% complement split in this repo.
export TEST_MODE="${TEST_MODE:-True}"

# Overridable run parameters (VSR-only).
DATASET="${DATASET:-VSR}"
MODEL_NAME="${MODEL_NAME:-llava1.5}"
METHOD="${METHOD:-base}"
WEIGHT="${WEIGHT:-1.0}"
WEIGHT1="${WEIGHT1:-0.5}"
WEIGHT2="${WEIGHT2:-1.2}"
THRESHOLD="${THRESHOLD:-0.64}"
SEED="${SEED:-1}"
DOWNLOAD_FLAG="${DOWNLOAD_FLAG:---download}"

if [ "$DATASET" != "VSR" ]; then
  echo "ERROR: DATASET must be VSR for this script. Got: $DATASET"
  exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
nvidia-smi || true

set -x
python -u main_aro.py \
  --dataset "$DATASET" \
  --model-name "$MODEL_NAME" \
  --method "$METHOD" \
  --weight "$WEIGHT" \
  --weight1 "$WEIGHT1" \
  --weight2 "$WEIGHT2" \
  --threshold "$THRESHOLD" \
  --seed "$SEED" \
  $DOWNLOAD_FLAG
