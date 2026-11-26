#!/bin/bash
# The 8-node data-efficiency sweep on the cluster: 2 history conditions x 4 fixed training-set
# sizes x 3 seeds = 24 runs, one array task each, with the default settings in config.py
# (8 nodes, train on 3 steps, test on 3-6, at most 4,000 steps). Each task also measures
# edge-attention share on its finished model.
#     sbatch --account=def-ssalihog_gpu scripts/cluster/sweep_8nodes_array.sh
#SBATCH --job-name=scratchpad-8n
#SBATCH --array=0-23
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
set -euo pipefail

KS=(1 0)                      # 0 means full history
SIZES=(250 1000 4000 16000)
SEEDS=(0 1 2)
RESULTS_DIR=${RESULTS_DIR:-results/_cluster_8nodes_3step}
MODEL_ARGS=${MODEL_ARGS:-}       # e.g. "--n-layers 6 --n-heads 6 --hidden-size 384"

I=$SLURM_ARRAY_TASK_ID
K=${KS[$((I % 2))]}
N=${SIZES[$((I / 2 % 4))]}
SEED=${SEEDS[$((I / 8))]}

source scripts/cluster/setup_env.sh   # virtualenv on the node-local disk
echo "k=$K n_train=$N seed=$SEED -> $RESULTS_DIR"

LOCAL_OUT="$SLURM_TMPDIR/out"
python -m scratchpad.train --k "$K" --n-train "$N" --seed "$SEED" --results-dir "$LOCAL_OUT" $MODEL_ARGS
python -m scratchpad.attention --run "$LOCAL_OUT"/k*_n"${N}"_s"${SEED}"
mkdir -p "$RESULTS_DIR" && cp -r "$LOCAL_OUT"/. "$RESULTS_DIR"/
