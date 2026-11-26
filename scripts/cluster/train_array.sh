#!/bin/bash
# One training run per array task: every history condition x seed, each on one GPU.
# Submit from the project folder (pass your own account; nothing is hard-coded here):
#     sbatch --account=def-ssalihog_gpu scripts/cluster/train_array.sh
# Override the task size or seeds without editing this file:
#     sbatch --account=def-ssalihog_gpu --export=ALL,N_NODES=24,SEEDS="0 1 2" scripts/cluster/train_array.sh
# The array size must equal (number of KS) x (number of SEEDS); change it with --array=0-N.
# Check the queue with `squeue -u $USER`; logs land in slurm_logs/.
#SBATCH --job-name=scratchpad
#SBATCH --array=0-11
# A whole H100 runs a step about 4.5x faster than a 10 GB slice (measured 2026-11-25), so a full
# 50,000-step run at 24 nodes takes about 1.5 h and fits the 3-hour queue class. Whole cards wait
# longer in the queue (about 50 min for one job that day). For a slice, which starts within minutes
# but needs a longer limit, add to the sbatch command:
#     --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
set -euo pipefail

N_NODES=${N_NODES:-16}
TRAIN_STEPS=${TRAIN_STEPS:-6}
TEST_STEPS=${TEST_STEPS:-6,8,10,12}
MAX_STEPS=${MAX_STEPS:-50000}
KS=(${KS:-0 4 2 1})          # 0 means full history
SEEDS=(${SEEDS:-0 1 2})
RESULTS_DIR=${RESULTS_DIR:-results/_cluster_${N_NODES}nodes_${TRAIN_STEPS}step_fresh}
# Model shape overrides, e.g. MODEL_ARGS="--n-layers 6 --n-heads 6 --hidden-size 384" (about 10.6M
# parameters, the layers/heads/width of Abbe et al. 2024). Empty means the 4.2M default in config.py.
MODEL_ARGS=${MODEL_ARGS:-}

K=${KS[$((SLURM_ARRAY_TASK_ID % ${#KS[@]}))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / ${#KS[@]}))]}

source scripts/cluster/setup_env.sh   # virtualenv on the node-local disk
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "k=$K seed=$SEED n_nodes=$N_NODES train_steps=$TRAIN_STEPS model_args=[$MODEL_ARGS] -> $RESULTS_DIR"

# Train into the node-local disk and copy the finished run back, so the slow network disk is
# written once. A run cut off by the time limit leaves nothing behind.
LOCAL_OUT="$SLURM_TMPDIR/out"
python -m scratchpad.train --k "$K" --seed "$SEED" --fresh \
    --n-nodes "$N_NODES" --train-steps "$TRAIN_STEPS" --test-steps "$TEST_STEPS" \
    --max-steps "$MAX_STEPS" --eval-every 500 --results-dir "$LOCAL_OUT" $MODEL_ARGS
mkdir -p "$RESULTS_DIR" && cp -r "$LOCAL_OUT"/. "$RESULTS_DIR"/
