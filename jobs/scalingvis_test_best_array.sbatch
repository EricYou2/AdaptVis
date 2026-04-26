#!/bin/bash
#SBATCH -J scalingvis.test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH -o logs/scalingvis-test-%A_%a.out
#SBATCH -e logs/scalingvis-test-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output outputs sweeps/scalingvis/raw/test

CONDA_BASE="/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate adaptvis_py311

export TEST_MODE=True
export ADAPTVIS_DYNAMIC_PREPROCESS="${ADAPTVIS_DYNAMIC_PREPROCESS:-True}"

DATASETS=(
  "Controlled_Images_A"
  "Controlled_Images_B"
  "COCO_QA_one_obj"
  "COCO_QA_two_obj"
  "VG_QA_one_obj"
  "VG_QA_two_obj"
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
BEST_JSON="sweeps/scalingvis/reports/best_weights.json"

if [[ ! -f "$BEST_JSON" ]]; then
  echo "[error] missing $BEST_JSON. Run jobs/scalingvis_val_plot.sbatch first."
  exit 1
fi

WEIGHT=$(python - <<'PY'
import json
from pathlib import Path
import os
p = Path("sweeps/scalingvis/reports/best_weights.json")
data = json.loads(p.read_text(encoding="utf-8"))
dataset = os.environ["DATASET"]
v = data.get(dataset, {}).get("weight")
print(v if v is not None else "")
PY
)

if [[ -z "$WEIGHT" ]]; then
  echo "[error] no best weight found for dataset=$DATASET in $BEST_JSON"
  exit 1
fi

OPTION="four"
if [[ "$DATASET" == "VG_QA_one_obj" || "$DATASET" == "VG_QA_two_obj" ]]; then
  OPTION="six"
fi

echo "[info] dataset=$DATASET best_weight=$WEIGHT option=$OPTION TEST_MODE=$TEST_MODE"

python -u main_aro.py \
  --dataset "$DATASET" \
  --model-name llava1.5 \
  --method scaling_vis \
  --weight "$WEIGHT" \
  --option "$OPTION" \
  --seed 1 \
  --download

mkdir -p "sweeps/scalingvis/raw/test/$DATASET"
SRC="output/results1.5_${DATASET}_scaling_vis_${WEIGHT}_${OPTION}option_Truescores.json"
DST="sweeps/scalingvis/raw/test/${DATASET}/best_weight_${WEIGHT}.json"

if [[ -f "$SRC" ]]; then
  cp "$SRC" "$DST"
  echo "[ok] copied $SRC -> $DST"
else
  echo "[warn] expected score file not found: $SRC"
fi
