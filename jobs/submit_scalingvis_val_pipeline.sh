#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

echo "[step] submit validation sweep array (7 datasets x 5 alphas = 35 jobs)"
VAL_OUT=$(sbatch --export=ALL,ADAPTVIS_DYNAMIC_PREPROCESS=${ADAPTVIS_DYNAMIC_PREPROCESS:-True} jobs/scalingvis_val_sweep_array.sbatch)
VAL_JOBID=$(echo "$VAL_OUT" | awk '{print $4}')
echo "[ok] val sweep job array id: $VAL_JOBID"

echo "[step] submit report/plot job after val sweep"
PLOT_OUT=$(sbatch --dependency=afterok:${VAL_JOBID} jobs/scalingvis_val_plot.sbatch)
PLOT_JOBID=$(echo "$PLOT_OUT" | awk '{print $4}')
echo "[ok] plot job id: $PLOT_JOBID"

echo "[step] submit test runs with best alpha after report is ready"
TEST_OUT=$(sbatch --dependency=afterok:${PLOT_JOBID} --export=ALL,ADAPTVIS_DYNAMIC_PREPROCESS=${ADAPTVIS_DYNAMIC_PREPROCESS:-True} jobs/scalingvis_test_best_array.sbatch)
TEST_JOBID=$(echo "$TEST_OUT" | awk '{print $4}')
echo "[ok] test sweep job array id: $TEST_JOBID"

echo ""
echo "Submitted pipeline:"
echo "  val array : $VAL_JOBID"
echo "  plot job  : $PLOT_JOBID"
echo "  test array: $TEST_JOBID"
echo ""
echo "Monitor with:"
echo "  squeue -j ${VAL_JOBID},${PLOT_JOBID},${TEST_JOBID}"
