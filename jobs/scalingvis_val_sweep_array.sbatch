#!/bin/bash
#SBATCH -J scalingvis.val
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH -o logs/scalingvis-val-%A_%a.out
#SBATCH -e logs/scalingvis-val-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output outputs sweeps/scalingvis/raw/val

CONDA_BASE="/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate adaptvis_py311

export TEST_MODE=False
export ADAPTVIS_DYNAMIC_PREPROCESS="${ADAPTVIS_DYNAMIC_PREPROCESS:-True}"

DATASETS=(
  "Controlled_Images_A"
  "Controlled_Images_B"
  "COCO_QA_one_obj"
  "COCO_QA_two_obj"
  "VG_QA_one_obj"
  "VG_QA_two_obj"
)
WEIGHTS=(0.5 0.8 1.2 1.5 2.0)

dataset_idx=$((SLURM_ARRAY_TASK_ID / ${#WEIGHTS[@]}))
weight_idx=$((SLURM_ARRAY_TASK_ID % ${#WEIGHTS[@]}))

DATASET="${DATASETS[$dataset_idx]}"
WEIGHT="${WEIGHTS[$weight_idx]}"

OPTION="four"
if [[ "$DATASET" == "VG_QA_one_obj" || "$DATASET" == "VG_QA_two_obj" ]]; then
  OPTION="six"
fi

echo "[info] dataset=$DATASET weight=$WEIGHT option=$OPTION TEST_MODE=$TEST_MODE"

python -u main_aro.py \
  --dataset "$DATASET" \
  --model-name llava1.5 \
  --method scaling_vis \
  --weight "$WEIGHT" \
  --option "$OPTION" \
  --seed 1 \
  --download

mkdir -p "sweeps/scalingvis/raw/val/$DATASET"
SRC="output/results1.5_${DATASET}_scaling_vis_${WEIGHT}_${OPTION}option_Falsescores.json"
DST="sweeps/scalingvis/raw/val/${DATASET}/weight_${WEIGHT}.json"

if [[ -f "$SRC" ]]; then
  cp "$SRC" "$DST"
  echo "[ok] copied $SRC -> $DST"
else
  echo "[warn] expected score file not found: $SRC"
fi
