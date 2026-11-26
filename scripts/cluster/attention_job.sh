#!/bin/bash
# Measure edge-attention share on every finished run in a results folder (writes attention.json
# into each run folder).
#     sbatch --account=def-ssalihog_gpu --export=ALL,RESULTS_DIR=results/_cluster_16nodes_6step_fresh scripts/cluster/attention_job.sh
#SBATCH --job-name=scratchpad-attn
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
set -euo pipefail
: "${RESULTS_DIR:?set RESULTS_DIR}"
source scripts/cluster/setup_env.sh
for run in "$RESULTS_DIR"/*/; do
    [ -f "$run/metrics.json" ] || continue
    echo "== $run"
    python -m scratchpad.attention --run "$run"
done
