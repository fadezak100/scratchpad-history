#!/bin/bash
# Short check that the cluster setup works: build the environment, run the tests, time 300 steps.
#     sbatch --account=def-ssalihog_gpu scripts/cluster/smoke_test.sh
#SBATCH --job-name=scratchpad-smoke
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x_%j.out
set -euo pipefail
time source scripts/cluster/setup_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pytest -q -p no:cacheprovider
time python -m scratchpad.train --k 1 --fresh --n-nodes 16 --train-steps 6 --test-steps 6,8,10,12 \
    --max-steps 300 --eval-every 100 --results-dir "$SLURM_TMPDIR/smoke"
